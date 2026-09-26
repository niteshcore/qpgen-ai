"""
Upload course material (PDF) and generate questions grounded in it — the
RAG half of question generation, distinct from ai_service's un-grounded
"generate from a topic string" path.

Reuses the 'questions.generate_ai' permission rather than adding a new one:
uploading/generating from documents is squarely part of the AI-generation
feature set, and every role that can already generate AI questions
(teacher, admin) is exactly who should be able to do this too.
"""
import os

from flask import Blueprint, request, jsonify, current_app

from app.extensions import db
from app.models.document import SourceDocument, DocumentChunk
from app.models.question import Question
from app.models.subject import Subject
from app.authorization import require_permission, get_current_user
from app.services import document_service, rag_service, embedding_service
from app.services.audit_service import log_action

documents_bp = Blueprint('documents', __name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


def _accessible_subject_ids(user):
    """None means "all subjects" (admin); otherwise the list a teacher may touch."""
    return None if user.has_role('admin') else [s.id for s in user.subjects]


def _check_subject_access(user, subject_id):
    allowed = _accessible_subject_ids(user)
    return allowed is None or subject_id in allowed


@documents_bp.route('/upload', methods=['POST'])
@require_permission('questions.generate_ai')
def upload_document():
    """Multipart upload: file, subject_id, optional title."""
    user = get_current_user()

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No selected file'}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400

    subject_id = request.form.get('subject_id', type=int)
    if not subject_id:
        return jsonify({'error': 'subject_id is required'}), 400
    if not db.session.get(Subject, subject_id):
        return jsonify({'error': 'Subject not found'}), 404
    if not _check_subject_access(user, subject_id):
        return jsonify({'error': 'Forbidden', 'message': 'You are not assigned to this subject'}), 403

    file_bytes = file.read(MAX_UPLOAD_BYTES + 1)
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        return jsonify({'error': f'File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB limit'}), 413

    try:
        document = document_service.create_document(
            subject_id=subject_id, uploaded_by=user.id, filename=file.filename,
            file_bytes=file_bytes, title=request.form.get('title'),
        )
    except document_service.DocumentProcessingError as e:
        return jsonify({'error': str(e)}), 422

    document_service.embed_document_background(current_app._get_current_object(), document.id)

    log_action('document.upload', resource_type='source_document', resource_id=document.id,
               details={'subject_id': subject_id, 'filename': file.filename, 'pages': document.page_count})

    return jsonify({'message': 'Document uploaded, processing in the background', 'document': document.to_dict()}), 201


@documents_bp.route('', methods=['GET'])
@require_permission('questions.read')
def list_documents():
    user = get_current_user()
    subject_id = request.args.get('subject_id', type=int)
    allowed = _accessible_subject_ids(user)

    query = SourceDocument.query
    if subject_id:
        if not _check_subject_access(user, subject_id):
            return jsonify({'error': 'Forbidden', 'message': 'You are not assigned to this subject'}), 403
        query = query.filter_by(subject_id=subject_id)
    elif allowed is not None:
        if not allowed:
            return jsonify({'documents': [], 'count': 0}), 200
        query = query.filter(SourceDocument.subject_id.in_(allowed))

    documents = query.order_by(SourceDocument.created_at.desc()).all()
    return jsonify({'documents': [d.to_dict() for d in documents], 'count': len(documents)}), 200


@documents_bp.route('/<int:document_id>', methods=['GET'])
@require_permission('questions.read')
def get_document(document_id):
    document = db.session.get(SourceDocument, document_id)
    if not document:
        return jsonify({'error': 'Document not found'}), 404
    if not _check_subject_access(get_current_user(), document.subject_id):
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify({'document': document.to_dict()}), 200


@documents_bp.route('/<int:document_id>', methods=['DELETE'])
@require_permission('questions.generate_ai')
def delete_document(document_id):
    document = db.session.get(SourceDocument, document_id)
    if not document:
        return jsonify({'error': 'Document not found'}), 404
    user = get_current_user()
    if not _check_subject_access(user, document.subject_id):
        return jsonify({'error': 'Forbidden'}), 403
    if document.uploaded_by != user.id and not user.has_role('admin'):
        return jsonify({'error': 'Forbidden', 'message': 'You do not own this document'}), 403

    db.session.delete(document)  # cascades to chunks; questions generated from it keep source_page, lose the FK (ON DELETE SET NULL)
    db.session.commit()
    log_action('document.delete', resource_type='source_document', resource_id=document_id)
    return jsonify({'message': 'Document deleted'}), 200


@documents_bp.route('/<int:document_id>/generate', methods=['POST'])
@require_permission('questions.generate_ai')
def generate_from_document(document_id):
    """
    Retrieves the chunks of `document_id` most relevant to `topic`, and asks
    Gemini to write questions grounded only in them. Always saves the
    result (matching teacher.generate_and_save_question's behaviour) with
    source_document_id/source_page set on each new Question.
    """
    document = db.session.get(SourceDocument, document_id)
    if not document:
        return jsonify({'error': 'Document not found'}), 404
    user = get_current_user()
    if not _check_subject_access(user, document.subject_id):
        return jsonify({'error': 'Forbidden'}), 403
    if document.status != 'ready':
        return jsonify({'error': f"Document is not ready yet (status: {document.status})",
                        'status': document.status}), 409

    data = request.get_json() or {}
    topic = (data.get('topic') or '').strip()
    if not topic:
        return jsonify({'error': 'topic is required — what should the questions focus on?'}), 400
    question_type = data.get('question_type', 'short')
    difficulty = data.get('difficulty', 'medium')
    marks = data.get('marks', 3)
    count = max(1, min(int(data.get('count', 3)), 10))

    try:
        chunks = rag_service.retrieve_chunks(document.subject_id, topic, document_id=document_id)
        generated = rag_service.generate_grounded_questions(
            subject_name=document.subject.name, chunks=chunks, question_type=question_type,
            difficulty=difficulty, marks=marks, count=count, focus=topic,
        )
    except rag_service.RagUnavailable as e:
        return jsonify({'error': str(e)}), 409
    except Exception as e:
        log_action('document.generate', status='failure', resource_type='source_document',
                   resource_id=document_id, details={'error': str(e)})
        return jsonify({'error': f'Grounded generation failed: {str(e)}'}), 500

    created = []
    for q in generated:
        question = Question(
            text=q['text'], question_type=q['question_type'], blooms_level=q['blooms_level'],
            difficulty=q['difficulty'], marks=q['marks'],
            option_a=q.get('option_a'), option_b=q.get('option_b'),
            option_c=q.get('option_c'), option_d=q.get('option_d'),
            correct_answer=q.get('correct_answer'),
            subject_id=document.subject_id, created_by=user.id,
            source_document_id=q.get('_source_document_id'), source_page=q.get('source_page'),
        )
        db.session.add(question)
        created.append(question)
    db.session.commit()

    embedding_service.index_questions_background(
        current_app._get_current_object(), [q.id for q in created]
    )

    log_action('document.generate', resource_type='source_document', resource_id=document_id,
               details={'topic': topic, 'count': len(created)})

    return jsonify({
        'message': f'Generated {len(created)} question(s) grounded in {document.filename}',
        'questions': [q.to_dict() for q in created],
    }), 201

"""
Public, unauthenticated routes — the community paper library.

Anyone can browse or download a paper here, no login required. Only papers
explicitly opted-in by their creator (Paper.is_public=True) are ever exposed
here; nothing behind auth leaks through this blueprint.
"""
from flask import Blueprint, request, jsonify, send_file
from app.models.paper import Paper

public_bp = Blueprint('public', __name__)


@public_bp.route('/papers', methods=['GET'])
def list_public_papers():
    """List shared papers, newest first. Optional ?subject_id= filter."""
    limit = min(request.args.get('limit', type=int, default=24), 100)
    offset = request.args.get('offset', type=int, default=0)
    subject_id = request.args.get('subject_id', type=int)

    query = Paper.query.filter_by(is_public=True)
    if subject_id:
        query = query.filter_by(subject_id=subject_id)
    query = query.order_by(Paper.created_at.desc())

    total = query.count()
    papers = query.offset(offset).limit(limit).all()

    return jsonify({
        'papers': [p.to_dict() for p in papers],
        'count': total,
    }), 200


@public_bp.route('/papers/<int:paper_id>', methods=['GET'])
def get_public_paper(paper_id):
    """Full detail (incl. questions) for one shared paper."""
    paper = Paper.query.filter_by(id=paper_id, is_public=True).first()
    if not paper:
        return jsonify({'error': 'Not found'}), 404

    paper_data = paper.to_dict()
    paper_data['questions'] = [q.to_dict() for q in paper.questions]
    return jsonify({'paper': paper_data}), 200


@public_bp.route('/papers/<int:paper_id>/pdf', methods=['GET'])
def download_public_paper_pdf(paper_id):
    """Stream the PDF for one shared paper — no auth, no rate limit beyond what the server already has."""
    from app.services.pdf_generator import generate_paper_pdf

    paper = Paper.query.filter_by(id=paper_id, is_public=True).first()
    if not paper:
        return jsonify({'error': 'Not found'}), 404

    paper_data = paper.to_dict()
    paper_data['questions'] = [q.to_dict() for q in paper.questions]
    paper_data['subject_name'] = paper.subject.name if paper.subject else 'Examination'
    paper_data['subject_code'] = paper.subject.code if paper.subject else ''

    pdf_buffer = generate_paper_pdf(paper_data)
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"{paper.title.replace(' ', '_')}.pdf",
    )

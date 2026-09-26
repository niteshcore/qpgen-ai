import io

import pytest
from reportlab.pdfgen import canvas

from app.extensions import db
from app.models.document import SourceDocument, DocumentChunk
from app.models.question import Question
from app.models.subject import Subject
from app.services import rag_service


def _make_pdf(pages):
    """Builds a small in-memory PDF; `pages` is a list of page body strings."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        c.setFont("Helvetica", 11)
        y = 700
        for line in text.strip().split("\n"):
            c.drawString(72, y, line)
            y -= 16
        c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()


BIOLOGY_PDF = _make_pdf([
    "Photosynthesis converts light energy into chemical energy in chloroplasts using chlorophyll.",
    "Cellular respiration breaks down glucose in mitochondria to produce ATP via the electron transport chain.",
])


def _upload(client, headers, subject_id, pdf_bytes=BIOLOGY_PDF, filename='notes.pdf', title=None):
    data = {'subject_id': str(subject_id), 'file': (io.BytesIO(pdf_bytes), filename)}
    if title:
        data['title'] = title
    return client.post('/api/documents/upload', headers=headers, data=data, content_type='multipart/form-data')


class FakeGemini:
    """Deterministic stand-in for rag_service._call_gemini, monkeypatched per test."""
    def __init__(self, questions):
        self.questions = questions

    def __call__(self, prompt, difficulty='medium'):
        import json
        return json.dumps(self.questions)


def test_upload_creates_document_and_embedded_chunks(app, client, auth_headers, init_database):
    sid = init_database['subject_id']
    res = _upload(client, auth_headers, sid, title='Biology Notes')
    assert res.status_code == 201
    doc = res.get_json()['document']
    assert doc['page_count'] == 2
    assert doc['chunk_count'] == 2
    assert doc['title'] == 'Biology Notes'
    # TESTING config runs embedding inline, so this is already 'ready' by the time the response returns.
    assert doc['status'] == 'ready'

    chunks = DocumentChunk.query.filter_by(document_id=doc['id']).order_by(DocumentChunk.chunk_index).all()
    assert [c.page_number for c in chunks] == [1, 2]
    assert all(c.embedding is not None for c in chunks)


def test_upload_rejects_non_pdf(app, client, auth_headers, init_database):
    data = {'subject_id': str(init_database['subject_id']), 'file': (io.BytesIO(b'hello'), 'notes.txt')}
    res = client.post('/api/documents/upload', headers=auth_headers, data=data, content_type='multipart/form-data')
    assert res.status_code == 400


def test_upload_rejects_pdf_with_no_extractable_text(app, client, auth_headers, init_database):
    blank_pdf = _make_pdf([''])  # a page with no text drawn on it
    res = _upload(client, auth_headers, init_database['subject_id'], pdf_bytes=blank_pdf)
    assert res.status_code == 422
    assert SourceDocument.query.count() == 0  # nothing half-created


def test_upload_denies_unassigned_subject(app, client, auth_headers, init_database):
    other = Subject(name='Other', code='OT1')
    db.session.add(other)
    db.session.commit()
    res = _upload(client, auth_headers, other.id)
    assert res.status_code == 403


def test_list_and_get_document(app, client, auth_headers, init_database):
    doc_id = _upload(client, auth_headers, init_database['subject_id']).get_json()['document']['id']

    res = client.get('/api/documents', headers=auth_headers)
    assert res.status_code == 200
    assert res.get_json()['count'] == 1

    res = client.get(f'/api/documents/{doc_id}', headers=auth_headers)
    assert res.status_code == 200
    assert res.get_json()['document']['id'] == doc_id


def test_delete_document_cascades_chunks_and_nulls_question_source(app, client, auth_headers, init_database, monkeypatch):
    sid = init_database['subject_id']
    doc_id = _upload(client, auth_headers, sid).get_json()['document']['id']

    monkeypatch.setattr(rag_service, '_call_gemini', FakeGemini([
        {'text': 'What pigment absorbs light in photosynthesis?', 'question_type': 'short',
         'blooms_level': 'remember', 'difficulty': 'easy', 'marks': 2,
         'correct_answer': 'Chlorophyll', 'source_passage': 1},
    ]))
    res = client.post(f'/api/documents/{doc_id}/generate', headers=auth_headers, json={'topic': 'photosynthesis'})
    qid = res.get_json()['questions'][0]['id']
    assert db.session.get(Question, qid).source_document_id == doc_id

    assert client.delete(f'/api/documents/{doc_id}', headers=auth_headers).status_code == 200
    assert DocumentChunk.query.filter_by(document_id=doc_id).count() == 0
    db.session.expire_all()
    q = db.session.get(Question, qid)
    assert q is not None  # the question itself survives
    assert q.source_document_id is None  # ON DELETE SET NULL, not cascaded


def test_generate_requires_ready_document(app, client, auth_headers, init_database):
    sid = init_database['subject_id']
    doc_id = _upload(client, auth_headers, sid).get_json()['document']['id']
    doc = db.session.get(SourceDocument, doc_id)
    doc.status = 'processing'
    db.session.commit()

    res = client.post(f'/api/documents/{doc_id}/generate', headers=auth_headers, json={'topic': 'photosynthesis'})
    assert res.status_code == 409


def test_generate_requires_topic(app, client, auth_headers, init_database):
    doc_id = _upload(client, auth_headers, init_database['subject_id']).get_json()['document']['id']
    res = client.post(f'/api/documents/{doc_id}/generate', headers=auth_headers, json={})
    assert res.status_code == 400


def test_generate_grounded_questions_cite_the_correct_page(app, client, auth_headers, init_database, monkeypatch):
    sid = init_database['subject_id']
    doc_id = _upload(client, auth_headers, sid).get_json()['document']['id']

    monkeypatch.setattr(rag_service, '_call_gemini', FakeGemini([
        {'text': 'Where does the Calvin cycle occur?', 'question_type': 'short', 'blooms_level': 'understand',
         'difficulty': 'medium', 'marks': 3, 'correct_answer': 'In the chloroplast stroma.', 'source_passage': 1},
        {'text': 'What produces most ATP in respiration?', 'question_type': 'short', 'blooms_level': 'understand',
         'difficulty': 'medium', 'marks': 3, 'correct_answer': 'The electron transport chain.', 'source_passage': 2},
    ]))

    res = client.post(f'/api/documents/{doc_id}/generate', headers=auth_headers,
                      json={'topic': 'photosynthesis and respiration', 'count': 2})
    assert res.status_code == 201
    questions = res.get_json()['questions']
    assert len(questions) == 2
    assert questions[0]['source_page'] == 1
    assert questions[1]['source_page'] == 2
    assert all(q['source_document_id'] == doc_id for q in questions)

    saved = Question.query.filter(Question.id.in_([q['id'] for q in questions])).all()
    assert {q.marks for q in saved} == {3}
    assert {q.subject_id for q in saved} == {sid}


def test_generate_drops_citation_for_invalid_source_passage(app, client, auth_headers, init_database, monkeypatch):
    sid = init_database['subject_id']
    doc_id = _upload(client, auth_headers, sid).get_json()['document']['id']

    monkeypatch.setattr(rag_service, '_call_gemini', FakeGemini([
        {'text': 'A question with a bogus citation', 'question_type': 'short', 'blooms_level': 'understand',
         'difficulty': 'medium', 'marks': 3, 'correct_answer': 'n/a', 'source_passage': 99},
    ]))
    res = client.post(f'/api/documents/{doc_id}/generate', headers=auth_headers, json={'topic': 'anything'})
    assert res.status_code == 201
    q = res.get_json()['questions'][0]
    assert q['source_page'] is None
    assert q['source_document_id'] is None  # question still exists and is saved, just uncited


def test_retrieve_chunks_ranks_by_relevance(app, client, auth_headers, init_database):
    sid = init_database['subject_id']
    _upload(client, auth_headers, sid)  # FakeEmbedder makes retrieval deterministic (bag-of-words hashing)

    results = rag_service.retrieve_chunks(sid, 'photosynthesis chlorophyll chloroplasts', top_k=2)
    assert len(results) == 2
    assert results[0].page_number == 1  # the photosynthesis-heavy chunk should rank first


def test_generate_unavailable_when_no_ready_documents(app, init_database):
    with pytest.raises(rag_service.RagUnavailable):
        rag_service.generate_grounded_questions('Biology', chunks=[], count=1)

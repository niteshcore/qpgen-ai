from app.extensions import db
from app.models.question import Question
from app.models.question_embedding import QuestionEmbedding
from app.models.subject import Subject
from app.services import embedding_service
from app.services.embedding_service import EmbeddingUnavailable

from tests.conftest import FakeEmbedder


def _add(client, headers, subject_id, text):
    return client.post('/api/teacher/questions', headers=headers, json={
        'subject_id': subject_id, 'text': text, 'question_type': 'short',
        'blooms_level': 'understand', 'difficulty': 'easy', 'marks': 2,
    })


class BrokenEmbedder:
    def embed(self, texts, task_type='SEMANTIC_SIMILARITY'):
        raise EmbeddingUnavailable('quota exceeded')


def test_new_question_is_embedded(app, client, auth_headers, init_database):
    res = _add(client, auth_headers, init_database['subject_id'], 'Explain deadlock conditions in operating systems')
    assert res.status_code == 201
    row = db.session.get(QuestionEmbedding, res.get_json()['data']['id'])
    assert row is not None
    assert row.embedding.shape == (QuestionEmbedding.DIM,)
    assert res.get_json()['similar_existing'] == []


def test_duplicate_is_flagged_on_add(app, client, auth_headers, init_database):
    sid = init_database['subject_id']
    _add(client, auth_headers, sid, 'Explain deadlock conditions in operating systems')
    res = _add(client, auth_headers, sid, 'Explain deadlock conditions in operating systems')

    assert res.status_code == 201  # warns, never blocks
    similar = res.get_json()['similar_existing']
    assert len(similar) == 1
    assert similar[0]['is_duplicate'] is True
    assert similar[0]['similarity'] >= embedding_service.DUPLICATE_THRESHOLD


def test_semantic_search_ranks_closest_first(app, client, auth_headers, init_database):
    sid = init_database['subject_id']
    _add(client, auth_headers, sid, 'Normalization removes redundancy from relational tables')
    _add(client, auth_headers, sid, 'Explain deadlock conditions in operating systems')

    res = client.post('/api/questions/similar', headers=auth_headers,
                      json={'text': 'deadlock conditions in operating systems'})
    assert res.status_code == 200
    matches = res.get_json()['matches']
    assert len(matches) == 2
    assert 'deadlock' in matches[0]['question']['text']
    assert matches[0]['similarity'] > matches[1]['similarity']


def test_search_excludes_unassigned_subjects(app, client, auth_headers, init_database):
    other = Subject(name='Other', code='OT1', description='')
    db.session.add(other)
    db.session.flush()
    hidden = Question(text='Explain deadlock conditions in operating systems', question_type='short',
                      blooms_level='understand', difficulty='easy', marks=2,
                      subject_id=other.id, created_by=init_database['user_id'])
    db.session.add(hidden)
    db.session.commit()
    embedding_service.index_questions([hidden])
    db.session.commit()

    res = client.post('/api/questions/similar', headers=auth_headers,
                      json={'text': 'Explain deadlock conditions in operating systems'})
    assert res.get_json()['matches'] == []

    res = client.post('/api/questions/similar', headers=auth_headers,
                      json={'text': 'deadlock', 'subject_id': other.id})
    assert res.status_code == 403


def test_question_saves_even_if_embedding_fails(app, client, auth_headers, init_database):
    embedding_service.set_embedder(BrokenEmbedder())
    res = _add(client, auth_headers, init_database['subject_id'], 'Explain paging')
    assert res.status_code == 201
    assert db.session.get(QuestionEmbedding, res.get_json()['data']['id']) is None


def test_search_returns_503_when_embeddings_unavailable(app, client, auth_headers, init_database):
    embedding_service.set_embedder(BrokenEmbedder())
    res = client.post('/api/questions/similar', headers=auth_headers, json={'text': 'paging'})
    assert res.status_code == 503


def test_reembeds_only_when_text_changes(app, client, auth_headers, init_database):
    qid = _add(client, auth_headers, init_database['subject_id'], 'Explain paging').get_json()['data']['id']
    calls_after_create = FakeEmbedder.calls

    client.put(f'/api/questions/{qid}', headers=auth_headers, json={'marks': 5})
    assert FakeEmbedder.calls == calls_after_create

    client.put(f'/api/questions/{qid}', headers=auth_headers, json={'text': 'Explain segmentation'})
    assert FakeEmbedder.calls == calls_after_create + 1


def test_similar_to_existing_excludes_itself(app, client, auth_headers, init_database):
    sid = init_database['subject_id']
    qid = _add(client, auth_headers, sid, 'Explain deadlock conditions in operating systems').get_json()['data']['id']
    _add(client, auth_headers, sid, 'Explain deadlock conditions in modern operating systems')

    res = client.get(f'/api/questions/{qid}/similar', headers=auth_headers)
    assert res.status_code == 200
    ids = [m['question']['id'] for m in res.get_json()['matches']]
    assert qid not in ids and len(ids) == 1


def test_deleting_question_removes_embedding(app, client, auth_headers, init_database):
    qid = _add(client, auth_headers, init_database['subject_id'], 'Explain paging').get_json()['data']['id']
    assert db.session.get(QuestionEmbedding, qid) is not None

    assert client.delete(f'/api/questions/{qid}', headers=auth_headers).status_code == 200
    db.session.expire_all()
    assert db.session.get(QuestionEmbedding, qid) is None

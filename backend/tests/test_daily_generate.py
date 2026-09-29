from datetime import date

import daily_generate as dg
from app.extensions import db
from app.models.paper import Paper
from app.models.question import Question
from app.models.subject import Subject
from app.models.user import User


class FakeAI:
    def __init__(self, texts):
        self.texts = texts
        self.calls = 0

    def generate_questions_batch(self, subject_name, topic, distribution):
        self.calls += 1
        return [{'text': t, 'question_type': 'short', 'blooms_level': 'apply', 'difficulty': 'medium',
                 'marks': 3, 'correct_answer': 'x'} for t in self.texts]


def _admin_and_subject(init_database):
    return db.session.get(User, init_database['user_id']), db.session.get(Subject, init_database['subject_id'])


def test_plan_is_deterministic_and_rotates_through_subjects():
    subjects = [Subject(id=i, name=n, code=str(i)) for i, n in enumerate(list(dg.TOPICS)[:5], start=1)]
    day = date(2026, 9, 27)
    assert [(s.id, t, sh[0]) for s, t, sh in dg.plan_for(subjects, 2, day)] == \
           [(s.id, t, sh[0]) for s, t, sh in dg.plan_for(subjects, 2, day)]

    seen = set()
    for offset in range(len(subjects)):
        for s, _, _ in dg.plan_for(subjects, 1, date.fromordinal(day.toordinal() + offset)):
            seen.add(s.id)
    assert seen == {s.id for s in subjects}  # every subject gets a turn


def test_plan_skips_subjects_without_topics():
    assert dg.plan_for([Subject(id=1, name='Underwater Basket Weaving', code='X')], 2, date.today()) == []


def test_generate_one_saves_questions_and_a_public_paper(app, init_database):
    from app.services import embedding_service
    from tests.conftest import FakeEmbedder
    embedding_service.set_embedder(FakeEmbedder())
    admin, subject = _admin_and_subject(init_database)
    ai = FakeAI([f'Distinct question number {i} about topic alpha{i}' for i in range(6)])

    dg.generate_one(ai, admin, subject, 'Topic A', dg.PAPER_SHAPES[1], date(2026, 9, 27), dry_run=False)

    paper = Paper.query.one()
    assert paper.is_public is True
    assert paper.total_marks == 18 and len(paper.questions) == 6
    assert Question.query.filter_by(subject_id=subject.id, topic='Topic A').count() == 6


def test_generate_one_is_idempotent_for_the_same_paper(app, init_database):
    from app.services import embedding_service
    from tests.conftest import FakeEmbedder
    embedding_service.set_embedder(FakeEmbedder())
    admin, subject = _admin_and_subject(init_database)
    ai = FakeAI([f'Another distinct question {i} regarding beta{i}' for i in range(6)])

    for _ in range(2):
        dg.generate_one(ai, admin, subject, 'Topic B', dg.PAPER_SHAPES[0], date(2026, 9, 27), dry_run=False)

    assert Paper.query.count() == 1
    assert ai.calls == 1  # the second run never even called Gemini


def test_duplicates_of_existing_questions_are_not_saved(app, init_database):
    from app.services import embedding_service
    from tests.conftest import FakeEmbedder
    embedding_service.set_embedder(FakeEmbedder())
    admin, subject = _admin_and_subject(init_database)
    texts = [f'Unique question {i} covering gamma{i}' for i in range(6)]
    dg.generate_one(FakeAI(texts), admin, subject, 'Topic C', dg.PAPER_SHAPES[1], date(2026, 9, 27), dry_run=False)
    before = Question.query.count()

    # Same text again under a different topic: every question is now a duplicate -> nothing new saved.
    dg.generate_one(FakeAI(texts), admin, subject, 'Topic D', dg.PAPER_SHAPES[1], date(2026, 9, 27), dry_run=False)

    assert Question.query.count() == before
    assert Paper.query.count() == 1


def test_dry_run_writes_nothing(app, init_database):
    admin, subject = _admin_and_subject(init_database)
    ai = FakeAI(['q'])
    dg.generate_one(ai, admin, subject, 'Topic E', dg.PAPER_SHAPES[0], date.today(), dry_run=True)
    assert ai.calls == 0 and Paper.query.count() == 0


def test_generate_one_exception_stops_the_run_but_not_the_job(app, init_database, monkeypatch):
    """A quota error during generate_one should surface to main()'s except-block, which treats
    it as expected rather than a real failure (see daily_generate.py's exit-code logic)."""
    from app.services import embedding_service
    from tests.conftest import FakeEmbedder
    embedding_service.set_embedder(FakeEmbedder())
    admin, subject = _admin_and_subject(init_database)

    class QuotaExceeded(FakeAI):
        def generate_questions_batch(self, subject_name, topic, distribution):
            raise ValueError('AI generation failed after 3 attempts. Last error: 429 You exceeded your current quota')

    import pytest
    with pytest.raises(ValueError, match='quota'):
        dg.generate_one(QuotaExceeded([]), admin, subject, 'Topic Q', dg.PAPER_SHAPES[0], date.today(), dry_run=False)
    assert Paper.query.count() == 0  # nothing half-written

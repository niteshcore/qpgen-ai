"""
Semantic embeddings for the question bank.

- Embeds question text with Gemini (`gemini-embedding-001`, free tier) at 768 dims.
- Stores one vector per question in `question_embeddings`
  (pgvector on PostgreSQL, JSON on SQLite).
- Powers semantic search and near-duplicate detection.

Embedding is always best-effort from the request path: if the API key is
missing or the call fails, the question is still saved and the backfill
script (`backfill_embeddings.py`) can catch it up later.
"""
import hashlib
import logging
import os
import time

import numpy as np
import sqlalchemy as sa

from app.extensions import db
from app.models.question import Question
from app.models.question_embedding import QuestionEmbedding, EmbeddingVector

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = 'models/gemini-embedding-001'
EMBEDDING_DIM = QuestionEmbedding.DIM
BATCH_SIZE = 100  # Gemini's max inputs per embed request

# Cosine similarity at/above which two questions are flagged as near-duplicates.
# Calibrated on the seeded bank (423 questions, 10k same-subject pairs, p99 = 0.872):
# at 0.90 short same-topic questions ("Explain X" vs "Explain Y") start getting flagged;
# at 0.92 every flagged pair was a genuine rewording. See backfill_embeddings.py --report.
DUPLICATE_THRESHOLD = float(os.getenv('DUPLICATE_THRESHOLD', 0.92))


class EmbeddingUnavailable(Exception):
    """Raised when embeddings can't be produced (no key, API down, quota)."""


class GeminiEmbedder:
    def __init__(self, api_key=None, max_attempts=3):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.max_attempts = max_attempts
        if not self.api_key:
            raise EmbeddingUnavailable('GOOGLE_API_KEY is not configured')
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self._genai = genai

    def embed(self, texts, task_type='SEMANTIC_SIMILARITY'):
        """Returns an (n, EMBEDDING_DIM) float32 array of L2-normalised vectors."""
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start:start + BATCH_SIZE], task_type))
        return _normalise(np.asarray(vectors, dtype=np.float32))

    def _embed_batch(self, batch, task_type):
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self._genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=batch,
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIM,
                )
                return result['embedding']
            except Exception as e:
                last_error = e
                # Free tier is rate-limited per minute; back off and retry.
                if attempt < self.max_attempts:
                    time.sleep(2 ** attempt)
        raise EmbeddingUnavailable(f'Embedding failed after {self.max_attempts} attempts: {last_error}')


def _normalise(matrix):
    # gemini-embedding-001 only returns unit vectors at its full 3072 dims;
    # truncated outputs must be normalised for dot product == cosine similarity.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = GeminiEmbedder()
    return _embedder


def set_embedder(embedder):
    """Swap the embedder (tests use a deterministic fake)."""
    global _embedder
    _embedder = embedder


def embed_text(text):
    return get_embedder().embed([text])[0]


def _content_hash(text):
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()


def index_questions(questions, vectors=None):
    """
    Upserts embeddings for the given questions, skipping any whose text hasn't
    changed since it was last embedded. Pass `vectors` (aligned with `questions`)
    to reuse embeddings already computed, e.g. by a duplicate check.
    Adds to the session; the caller commits. Returns the number embedded.
    """
    pending = []
    for i, q in enumerate(questions):
        h = _content_hash(q.text)
        row = q.embedding_row
        if row is not None and row.content_hash == h and row.model == EMBEDDING_MODEL:
            continue
        pending.append((q, h, None if vectors is None else vectors[i]))

    if not pending:
        return 0

    missing = [p for p in pending if p[2] is None]
    if missing:
        fresh = get_embedder().embed([q.text for q, _, _ in missing])
        fresh_by_id = {id(q): v for (q, _, _), v in zip(missing, fresh)}
        pending = [(q, h, v if v is not None else fresh_by_id[id(q)]) for q, h, v in pending]

    for q, h, vec in pending:
        if q.embedding_row is None:
            q.embedding_row = QuestionEmbedding(model=EMBEDDING_MODEL, content_hash=h, embedding=vec)
        else:
            q.embedding_row.model = EMBEDDING_MODEL
            q.embedding_row.content_hash = h
            q.embedding_row.embedding = vec
    return len(pending)


def index_questions_best_effort(questions, vectors=None):
    """
    index_questions + commit, never raising. Returns True if embeddings were stored.

    On failure, the affected questions stay unembedded (invisible to semantic
    search / duplicate detection) until someone reruns backfill_embeddings.py —
    there's no automatic retry. Logged at WARNING for the expected case (no
    API key / quota, see EmbeddingUnavailable) and ERROR with a traceback for
    anything else, so an operator watching logs can tell "known, ignorable"
    apart from "investigate this."
    """
    try:
        index_questions(questions, vectors)
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        ids = [q.id for q in questions if q.id is not None]
        log = logger.warning if isinstance(e, EmbeddingUnavailable) else logger.error
        log('Embedding skipped for question(s) %s: %s', ids, e, exc_info=not isinstance(e, EmbeddingUnavailable))
        return False


def find_similar(vector, subject_ids=None, limit=5, exclude_ids=(), min_score=0.0):
    """
    Returns [(Question, cosine_similarity)] most similar first.
    `subject_ids=None` searches every subject (admins); otherwise restricts to those ids.
    Uses pgvector's cosine-distance operator on PostgreSQL, numpy on SQLite.
    """
    if subject_ids is not None and not subject_ids:
        return []

    if db.engine.dialect.name == 'postgresql':
        query_vec = sa.bindparam('query_vec', value=list(map(float, vector)), type_=EmbeddingVector(EMBEDDING_DIM))
        distance = QuestionEmbedding.embedding.op('<=>', return_type=sa.Float)(query_vec)
        query = (db.session.query(Question, distance.label('distance'))
                 .join(QuestionEmbedding, QuestionEmbedding.question_id == Question.id))
        if subject_ids is not None:
            query = query.filter(Question.subject_id.in_(subject_ids))
        if exclude_ids:
            query = query.filter(~Question.id.in_(list(exclude_ids)))
        rows = query.order_by(distance).limit(limit).all()
        results = [(q, 1.0 - float(d)) for q, d in rows]
    else:
        # Select the entity rather than the column: numpy arrays aren't hashable,
        # which breaks the ORM's row de-duplication.
        query = (db.session.query(Question, QuestionEmbedding)
                 .join(QuestionEmbedding, QuestionEmbedding.question_id == Question.id))
        if subject_ids is not None:
            query = query.filter(Question.subject_id.in_(subject_ids))
        if exclude_ids:
            query = query.filter(~Question.id.in_(list(exclude_ids)))
        rows = query.all()
        if not rows:
            return []
        matrix = np.stack([row.embedding for _, row in rows])
        scores = matrix @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores)[:limit]
        results = [(rows[i][0], float(scores[i])) for i in order]

    return [(q, s) for q, s in results if s >= min_score]


def check_duplicates_best_effort(text, subject_ids=None, limit=3):
    """
    Embeds `text` and returns (vector, near-duplicate matches), never raising.
    The vector is returned so the caller can store it without a second API call.
    On failure returns (None, []).
    """
    try:
        vector = embed_text(text)
        matches = find_similar(vector, subject_ids=subject_ids, limit=limit, min_score=DUPLICATE_THRESHOLD)
        return vector, matches
    except Exception as e:
        log = logger.warning if isinstance(e, EmbeddingUnavailable) else logger.error
        log('Duplicate check skipped: %s', e, exc_info=not isinstance(e, EmbeddingUnavailable))
        return None, []


def similar_to_dicts(matches):
    return [{
        'question': q.to_dict(),
        'similarity': round(score, 4),
        'is_duplicate': score >= DUPLICATE_THRESHOLD,
    } for q, score in matches]

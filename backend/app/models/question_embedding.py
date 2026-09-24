import json
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.types import TypeDecorator, Text

from app.extensions import db


class EmbeddingVector(TypeDecorator):
    """
    Stores an embedding as a pgvector `vector(dim)` on PostgreSQL (Supabase),
    and as a JSON-encoded list on SQLite (local dev / tests).
    Always hands back a float32 numpy array.
    """
    impl = Text
    cache_ok = True

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            from pgvector.sqlalchemy import Vector
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        values = [float(x) for x in value]
        if dialect.name == 'postgresql':
            return values  # pgvector's own bind processor serialises this
        return json.dumps(values)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, str):
            value = json.loads(value)
        return np.asarray(value, dtype=np.float32)


class QuestionEmbedding(db.Model):
    """One semantic embedding per question, used for similarity search and duplicate detection."""
    __tablename__ = 'question_embeddings'

    DIM = 768

    question_id = db.Column(db.Integer, db.ForeignKey('questions.id', ondelete='CASCADE'), primary_key=True)
    model = db.Column(db.String(64), nullable=False)
    content_hash = db.Column(db.String(64), nullable=False)  # sha256 of the embedded text; skip re-embedding if unchanged
    embedding = db.Column(EmbeddingVector(DIM), nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    # delete-orphan so deleting a Question through the ORM also removes its embedding
    # (SQLite doesn't enforce the ON DELETE CASCADE above by default).
    question = db.relationship(
        'Question',
        backref=db.backref('embedding_row', uselist=False, cascade='all, delete-orphan'),
    )

    def __repr__(self):
        return f'<QuestionEmbedding q={self.question_id} model={self.model}>'

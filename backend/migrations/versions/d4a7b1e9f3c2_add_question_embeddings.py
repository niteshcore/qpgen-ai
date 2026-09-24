"""add question_embeddings table (pgvector on PostgreSQL)

Revision ID: d4a7b1e9f3c2
Revises: c3e9a5f0d7b2
Create Date: 2026-09-25 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'd4a7b1e9f3c2'
down_revision = 'c3e9a5f0d7b2'
branch_labels = None
depends_on = None

EMBEDDING_DIM = 768


def upgrade():
    bind = op.get_bind()
    # Local SQLite dev DBs get this table from db.create_all() on startup.
    if 'question_embeddings' in sa.inspect(bind).get_table_names():
        return

    is_postgres = bind.dialect.name == 'postgresql'
    if is_postgres:
        from pgvector.sqlalchemy import Vector
        op.execute('CREATE EXTENSION IF NOT EXISTS vector')
        embedding_type = Vector(EMBEDDING_DIM)
    else:
        embedding_type = sa.Text()

    op.create_table(
        'question_embeddings',
        sa.Column('question_id', sa.Integer(), sa.ForeignKey('questions.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('model', sa.String(length=64), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('embedding', embedding_type, nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    )

    if is_postgres:
        # HNSW index for fast approximate cosine-distance search.
        op.execute(
            'CREATE INDEX ix_question_embeddings_hnsw ON question_embeddings '
            'USING hnsw (embedding vector_cosine_ops)'
        )


def downgrade():
    op.drop_table('question_embeddings')

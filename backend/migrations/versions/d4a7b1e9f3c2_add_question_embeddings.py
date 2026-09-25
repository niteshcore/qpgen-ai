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


def _table_exists_in_current_schema(bind, table_name):
    """
    Schema-scoped existence check. Inspector.get_table_names() without a
    schema= argument uses pg_table_is_visible(), which matches a table
    visible through ANY schema on search_path — not just the one DDL
    actually targets. That false-positives whenever another schema on the
    path (e.g. a shared database with multiple app environments) happens to
    already have a same-named table, silently skipping this migration.
    """
    if bind.dialect.name != 'postgresql':
        return table_name in sa.inspect(bind).get_table_names()
    return bind.execute(sa.text(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = current_schema() AND table_name = :name"
    ), {'name': table_name}).first() is not None


def upgrade():
    bind = op.get_bind()
    # Local SQLite dev DBs get this table from db.create_all() on startup.
    if _table_exists_in_current_schema(bind, 'question_embeddings'):
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

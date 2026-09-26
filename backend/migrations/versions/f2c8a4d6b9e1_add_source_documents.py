"""add source_documents + document_chunks (RAG grounding), question provenance columns

Revision ID: f2c8a4d6b9e1
Revises: d4a7b1e9f3c2
Create Date: 2026-09-26 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'f2c8a4d6b9e1'
down_revision = 'd4a7b1e9f3c2'
branch_labels = None
depends_on = None

CHUNK_EMBEDDING_DIM = 768


def upgrade():
    bind = op.get_bind()
    is_postgres = bind.dialect.name == 'postgresql'
    embedding_type = None
    if is_postgres:
        from pgvector.sqlalchemy import Vector
        op.execute('CREATE EXTENSION IF NOT EXISTS vector')  # no-op if d4a7b1e9f3c2 already ran it
        embedding_type = Vector(CHUNK_EMBEDDING_DIM)
    else:
        embedding_type = sa.Text()

    op.create_table(
        'source_documents',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('subject_id', sa.Integer(), nullable=False),
        sa.Column('uploaded_by', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=True),
        sa.Column('page_count', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['subject_id'], ['subjects.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['uploaded_by'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'document_chunks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', embedding_type, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['document_id'], ['source_documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    if is_postgres:
        op.execute(
            'CREATE INDEX ix_document_chunks_hnsw ON document_chunks '
            'USING hnsw (embedding vector_cosine_ops)'
        )

    with op.batch_alter_table('questions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('source_document_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('source_page', sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            'fk_questions_source_document', 'source_documents',
            ['source_document_id'], ['id'], ondelete='SET NULL',
        )


def downgrade():
    with op.batch_alter_table('questions', schema=None) as batch_op:
        batch_op.drop_constraint('fk_questions_source_document', type_='foreignkey')
        batch_op.drop_column('source_page')
        batch_op.drop_column('source_document_id')
    op.drop_table('document_chunks')
    op.drop_table('source_documents')

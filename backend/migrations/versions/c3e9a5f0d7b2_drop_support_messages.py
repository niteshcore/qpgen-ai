"""drop support_messages table (support ticket feature removed)

Revision ID: c3e9a5f0d7b2
Revises: b7d2f4a8e1c6
Create Date: 2026-09-07 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'c3e9a5f0d7b2'
down_revision = 'b7d2f4a8e1c6'
branch_labels = None
depends_on = None


def _table_exists_in_current_schema(bind, table_name):
    # Schema-scoped on purpose — see the same helper in
    # d4a7b1e9f3c2_add_question_embeddings.py for why get_table_names()
    # without a schema= argument isn't safe to use here.
    if bind.dialect.name != 'postgresql':
        return table_name in sa.inspect(bind).get_table_names()
    return bind.execute(sa.text(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = current_schema() AND table_name = :name"
    ), {'name': table_name}).first() is not None


def upgrade():
    # support_messages, like several other tables in this project's early history,
    # was only ever created via db.create_all() on local SQLite (see app/__init__.py) —
    # it has no CREATE TABLE migration, so a fresh Postgres database reaching this
    # point never has it. Guard the drop so `flask db upgrade` works from empty.
    bind = op.get_bind()
    if _table_exists_in_current_schema(bind, 'support_messages'):
        op.drop_table('support_messages')


def downgrade():
    op.create_table(
        'support_messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('subject', sa.String(length=200), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )

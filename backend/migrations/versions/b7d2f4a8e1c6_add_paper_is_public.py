"""add is_public flag to papers (opt-in public library)

Revision ID: b7d2f4a8e1c6
Revises: a1f3e7c9d2b4
Create Date: 2026-09-07 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'b7d2f4a8e1c6'
down_revision = 'a1f3e7c9d2b4'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('papers', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('is_public', sa.Boolean(), nullable=False, server_default=sa.false())
        )
    # server_default was only needed to backfill existing rows; drop it so the
    # ORM's Python-side default is the single source of truth going forward.
    with op.batch_alter_table('papers', schema=None) as batch_op:
        batch_op.alter_column('is_public', server_default=None)


def downgrade():
    with op.batch_alter_table('papers', schema=None) as batch_op:
        batch_op.drop_column('is_public')

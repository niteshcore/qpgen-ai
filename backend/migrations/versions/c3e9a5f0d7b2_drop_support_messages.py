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


def upgrade():
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

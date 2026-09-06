"""add open 'user' role for public self-serve paper generation

Registration used to grant the 'teacher' role, which is gated to subjects an
admin explicitly assigns. This adds a plain 'user' role that can read every
subject/question and generate/download its own papers, but cannot create,
edit, or delete anything in the question bank (that stays admin-only).

Revision ID: a1f3e7c9d2b4
Revises: c8910b936d04
Create Date: 2026-09-06 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime, timezone

# revision identifiers, used by Alembic.
revision = 'a1f3e7c9d2b4'
down_revision = 'c8910b936d04'
branch_labels = None
depends_on = None


USER_ROLE_PERMISSIONS = [
    'subjects.read',
    'questions.read',
    'papers.create',
    'papers.read',
    'papers.update',
    'papers.delete',
    'papers.download_pdf',
    'users.read_self',
]


def upgrade():
    bind = op.get_bind()

    roles = sa.table(
        'roles',
        sa.column('id', sa.Integer),
        sa.column('name', sa.String),
        sa.column('description', sa.String),
        sa.column('created_at', sa.DateTime),
    )
    permissions = sa.table(
        'permissions',
        sa.column('id', sa.Integer),
        sa.column('name', sa.String),
    )
    role_permissions = sa.table(
        'role_permissions',
        sa.column('role_id', sa.Integer),
        sa.column('permission_id', sa.Integer),
    )

    # Idempotent: skip if this migration already ran against this database.
    existing = bind.execute(sa.select(roles.c.id).where(roles.c.name == 'user')).first()
    if existing:
        return

    bind.execute(
        roles.insert().values(
            name='user',
            description=(
                'Open self-serve account — browse every subject and question, '
                'generate/download papers. Cannot edit the question bank.'
            ),
            created_at=datetime.now(timezone.utc),
        )
    )
    user_role_id = bind.execute(
        sa.select(roles.c.id).where(roles.c.name == 'user')
    ).scalar_one()

    perm_rows = bind.execute(
        sa.select(permissions.c.id).where(permissions.c.name.in_(USER_ROLE_PERMISSIONS))
    ).fetchall()

    if perm_rows:
        bind.execute(
            role_permissions.insert(),
            [{'role_id': user_role_id, 'permission_id': p.id} for p in perm_rows],
        )


def downgrade():
    bind = op.get_bind()

    roles = sa.table('roles', sa.column('id', sa.Integer), sa.column('name', sa.String))
    role_permissions = sa.table('role_permissions', sa.column('role_id', sa.Integer))

    row = bind.execute(sa.select(roles.c.id).where(roles.c.name == 'user')).first()
    if row:
        bind.execute(role_permissions.delete().where(role_permissions.c.role_id == row.id))
        bind.execute(roles.delete().where(roles.c.id == row.id))

"""add_zone_to_users

Revision ID: fb49810c430b
Revises: 8470b88af484
Create Date: 2026-05-14 15:37:13.423677

"""
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

from alembic import op

# revision identifiers, used by Alembic.
revision = 'fb49810c430b'
down_revision = '8470b88af484'
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    insp = Inspector.from_engine(bind)
    return any(c["name"] == column for c in insp.get_columns(table))


def _index_exists(index_name: str, table: str) -> bool:
    bind = op.get_bind()
    insp = Inspector.from_engine(bind)
    return any(ix["name"] == index_name for ix in insp.get_indexes(table))


def upgrade() -> None:
    # Add zone column to users (nullable, so existing rows default to NULL)
    if not _column_exists('users', 'zone'):
        op.add_column('users', sa.Column('zone', sa.String(length=100), nullable=True))

    if not _index_exists('ix_users_zone', 'users'):
        op.create_index('ix_users_zone', 'users', ['zone'], unique=False)


def downgrade() -> None:
    if _index_exists('ix_users_zone', 'users'):
        op.drop_index('ix_users_zone', table_name='users')

    if _column_exists('users', 'zone'):
        op.drop_column('users', 'zone')

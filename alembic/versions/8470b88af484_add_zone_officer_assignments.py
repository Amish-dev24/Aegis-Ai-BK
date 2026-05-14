"""add_zone_officer_assignments

Revision ID: 8470b88af484
Revises: c2d3e4f5a6b7
Create Date: 2026-05-14 15:32:19.946762

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector


# revision identifiers, used by Alembic.
revision = '8470b88af484'
down_revision = 'c2d3e4f5a6b7'
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    insp = Inspector.from_engine(bind)
    return table_name in insp.get_table_names()


def upgrade() -> None:
    # Create zone_officer_assignments only if it does not already exist
    # (SQLAlchemy may have auto-created it on app startup before this migration ran).
    if not _table_exists('zone_officer_assignments'):
        op.create_table(
            'zone_officer_assignments',
            sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column('company_id', sa.Integer(), sa.ForeignKey('companies.id', ondelete='CASCADE'), nullable=False, index=True),
            sa.Column('zone_name', sa.String(length=100), nullable=False),
            sa.Column('officer_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
            sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now()),
            sa.UniqueConstraint('company_id', 'zone_name', name='uq_zone_company'),
        )


def downgrade() -> None:
    if _table_exists('zone_officer_assignments'):
        op.drop_table('zone_officer_assignments')

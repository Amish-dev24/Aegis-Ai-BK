"""add detection settings tables

Revision ID: b1c2d3e4f5a6
Revises: ae3fd49977d7
Create Date: 2026-03-06

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b1c2d3e4f5a6'
down_revision = 'ae3fd49977d7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'global_module_settings',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('module_name', sa.String(50), nullable=False, unique=True, index=True),
        sa.Column('is_enabled', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('updated_by', sa.Integer(), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        'company_detection_settings',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('company_id', sa.Integer(), sa.ForeignKey('companies.id'), nullable=False),
        sa.Column('module_name', sa.String(50), nullable=False),
        sa.Column('is_enabled', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('critical_threshold', sa.Float(), nullable=True),
        sa.Column('high_threshold', sa.Float(), nullable=True),
        sa.Column('medium_threshold', sa.Float(), nullable=True),
        sa.Column('min_confidence', sa.Float(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint('company_id', 'module_name', name='uq_company_module'),
    )


def downgrade() -> None:
    op.drop_table('company_detection_settings')
    op.drop_table('global_module_settings')

"""add resolved_by to alerts

Revision ID: d1e2f3a4b5c6
Revises: fb49810c430b
Create Date: 2026-05-14

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "d1e2f3a4b5c6"
down_revision = "fb49810c430b"
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    insp = inspect(bind)
    return any(c["name"] == column for c in insp.get_columns(table))


def upgrade() -> None:
    if not _column_exists("alerts", "resolved_by"):
        op.add_column(
            "alerts",
            sa.Column("resolved_by", sa.String(100), nullable=True),
        )


def downgrade() -> None:
    if _column_exists("alerts", "resolved_by"):
        op.drop_column("alerts", "resolved_by")

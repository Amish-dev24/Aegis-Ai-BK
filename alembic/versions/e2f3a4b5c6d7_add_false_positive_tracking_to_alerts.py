"""add false_positive tracking to alerts

Revision ID: e2f3a4b5c6d7
Revises: d1e2f3a4b5c6
Create Date: 2026-05-14

"""
import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision = "e2f3a4b5c6d7"
down_revision = "d1e2f3a4b5c6"
branch_labels = None
depends_on = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    insp = inspect(bind)
    return any(c["name"] == column for c in insp.get_columns(table))


def upgrade() -> None:
    if not _column_exists("alerts", "false_positive_by"):
        op.add_column(
            "alerts",
            sa.Column("false_positive_by", sa.String(100), nullable=True),
        )
    if not _column_exists("alerts", "false_positive_at"):
        op.add_column(
            "alerts",
            sa.Column("false_positive_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    if _column_exists("alerts", "false_positive_at"):
        op.drop_column("alerts", "false_positive_at")
    if _column_exists("alerts", "false_positive_by"):
        op.drop_column("alerts", "false_positive_by")

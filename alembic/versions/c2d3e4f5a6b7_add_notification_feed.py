"""Add notification_feed for cross-user policy broadcasts.

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6

"""

from alembic import op
import sqlalchemy as sa


revision = "c2d3e4f5a6b7"
down_revision = "b1c2d3e4f5a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "notification_feed",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("audience", sa.String(length=20), nullable=False, index=True),
        sa.Column("company_id", sa.Integer(), sa.ForeignKey("companies.id"), nullable=True, index=True),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("actor_username", sa.String(length=150), nullable=True),
        sa.Column("actor_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
            index=True,
        ),
    )


def downgrade() -> None:
    op.drop_table("notification_feed")

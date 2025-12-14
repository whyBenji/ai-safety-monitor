"""add prompt payload to moderation results

Revision ID: f13c97550b8f
Revises: 
Create Date: 2025-11-17 00:47:50.155521

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f13c97550b8f'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column(
        "moderation_results",
        sa.Column("prompt_payload", sa.JSON(), nullable=True),
    )

def downgrade():
    op.drop_column("moderation_results", "prompt_payload")

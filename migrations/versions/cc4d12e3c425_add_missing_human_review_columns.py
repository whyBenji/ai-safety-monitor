"""add_missing_human_review_columns

Revision ID: cc4d12e3c425
Revises: 391610980da5
Create Date: 2025-12-14 05:57:44.892984

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cc4d12e3c425'
down_revision: Union[str, Sequence[str], None] = '391610980da5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add missing human review columns that should have been in the original schema."""
    # Check if columns already exist (they might exist in some databases)
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {col['name'] for col in inspector.get_columns('moderation_results')}
    
    # Add human_label if missing
    if 'human_label' not in existing_columns:
        op.add_column('moderation_results', sa.Column('human_label', sa.String(length=32), nullable=True))
    
    # Add human_notes if missing
    if 'human_notes' not in existing_columns:
        op.add_column('moderation_results', sa.Column('human_notes', sa.Text(), nullable=True))
    
    # Add human_reviewed_at if missing
    if 'human_reviewed_at' not in existing_columns:
        op.add_column('moderation_results', sa.Column('human_reviewed_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Remove human review columns."""
    # Only drop if they exist (safe to call even if already dropped)
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {col['name'] for col in inspector.get_columns('moderation_results')}
    
    if 'human_reviewed_at' in existing_columns:
        op.drop_column('moderation_results', 'human_reviewed_at')
    if 'human_notes' in existing_columns:
        op.drop_column('moderation_results', 'human_notes')
    if 'human_label' in existing_columns:
        op.drop_column('moderation_results', 'human_label')

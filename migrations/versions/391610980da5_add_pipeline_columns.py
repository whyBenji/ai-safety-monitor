"""add_pipeline_columns

Revision ID: 391610980da5
Revises: f13c97550b8f
Create Date: 2025-12-14 05:48:28.030692

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '391610980da5'
down_revision: Union[str, Sequence[str], None] = 'f13c97550b8f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add new columns for pipeline support (input/output classification, answer generation)."""
    # Get the bind to check database type
    bind = op.get_bind()
    is_postgres = bind.dialect.name == 'postgresql'
    
    # Add new columns to moderation_results table
    # Use server_default=False for boolean (works for both SQLite and PostgreSQL)
    op.add_column('moderation_results', sa.Column('input_flagged', sa.Boolean(), nullable=True))
    op.add_column('moderation_results', sa.Column('input_raw_response', sa.JSON(), nullable=True))
    op.add_column('moderation_results', sa.Column('answer_text', sa.Text(), nullable=True))
    op.add_column('moderation_results', sa.Column('answer_model', sa.String(length=255), nullable=True))
    op.add_column('moderation_results', sa.Column('answer_raw_response', sa.JSON(), nullable=True))
    op.add_column('moderation_results', sa.Column('output_flagged', sa.Boolean(), nullable=True))
    op.add_column('moderation_results', sa.Column('output_raw_response', sa.JSON(), nullable=True))
    op.add_column('moderation_results', sa.Column('human_label_type', sa.String(length=32), nullable=True))
    
    # Migrate existing data: copy flagged -> input_flagged, raw_response -> input_raw_response
    op.execute(sa.text("""
        UPDATE moderation_results 
        SET input_flagged = flagged 
        WHERE input_flagged IS NULL
    """))
    
    op.execute(sa.text("""
        UPDATE moderation_results 
        SET input_raw_response = raw_response 
        WHERE input_raw_response IS NULL
    """))
    
    # Set default values for input_flagged where still NULL
    # Use proper boolean syntax based on database type
    if is_postgres:
        op.execute(sa.text("""
            UPDATE moderation_results 
            SET input_flagged = FALSE 
            WHERE input_flagged IS NULL
        """))
    else:
        # SQLite
        op.execute(sa.text("""
            UPDATE moderation_results 
            SET input_flagged = 0 
            WHERE input_flagged IS NULL
        """))
    
    # Set default for input_raw_response where still NULL (empty JSON)
    op.execute(sa.text("""
        UPDATE moderation_results 
        SET input_raw_response = '{}' 
        WHERE input_raw_response IS NULL
    """))
    
    # Make input_flagged NOT NULL after data migration
    # Note: SQLite doesn't support ALTER COLUMN to change NOT NULL, so we skip this for SQLite
    # The columns will work fine with nullable=True in SQLite
    if is_postgres:
        op.alter_column('moderation_results', 'input_flagged', nullable=False, server_default=sa.text('FALSE'))
    
    # Add flag_type column to moderation_flags table
    op.add_column('moderation_flags', sa.Column('flag_type', sa.String(length=32), nullable=True, server_default='input'))
    
    # Set default value for existing flags
    op.execute(sa.text("""
        UPDATE moderation_flags 
        SET flag_type = 'input' 
        WHERE flag_type IS NULL
    """))
    
    # Make flag_type NOT NULL after data migration (SQLite limitation: can't change NOT NULL)
    if is_postgres:
        op.alter_column('moderation_flags', 'flag_type', nullable=False, server_default='input')


def downgrade() -> None:
    """Remove pipeline columns."""
    # Remove columns from moderation_flags
    op.drop_column('moderation_flags', 'flag_type')
    
    # Remove columns from moderation_results
    op.drop_column('moderation_results', 'human_label_type')
    op.drop_column('moderation_results', 'output_raw_response')
    op.drop_column('moderation_results', 'output_flagged')
    op.drop_column('moderation_results', 'answer_raw_response')
    op.drop_column('moderation_results', 'answer_model')
    op.drop_column('moderation_results', 'answer_text')
    op.drop_column('moderation_results', 'input_raw_response')
    op.drop_column('moderation_results', 'input_flagged')

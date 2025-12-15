"""restore_memory_indexes

Revision ID: ee60550ba239
Revises: 768c8f182893
Create Date: 2025-12-14 18:59:01.275062

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ee60550ba239'
down_revision: Union[str, Sequence[str], None] = '768c8f182893'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Restore indexes that were accidentally dropped in 768c8f182893."""
    # Vector similarity index (critical for retrieval performance)
    op.execute("""
        CREATE INDEX ix_memories_embedding ON memories
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)

    # Recency scoring index
    op.create_index('ix_memories_last_accessed_at', 'memories', ['last_accessed_at'])

    # Memory type filtering index
    op.create_index('ix_memories_memory_type', 'memories', ['memory_type'])

    # Composite index for user queries sorted by time
    op.execute("""
        CREATE INDEX ix_memories_created_at ON memories (user_id, created_at DESC)
    """)


def downgrade() -> None:
    """Drop the restored indexes."""
    op.drop_index('ix_memories_created_at', table_name='memories')
    op.drop_index('ix_memories_memory_type', table_name='memories')
    op.drop_index('ix_memories_last_accessed_at', table_name='memories')
    op.drop_index('ix_memories_embedding', table_name='memories')

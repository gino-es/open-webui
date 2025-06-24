"""Add chat embedding table

Revision ID: f75cfb55a211
Revises: 9f0c9cd09105
Create Date: 2025-06-24 14:49:10.406878

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f75cfb55a211'
down_revision: Union[str, None] = '9f0c9cd09105'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Get database connection to check dialect
    connection = op.get_bind()
    dialect = connection.dialect.name
    
    if dialect == 'postgresql':

    # Create chat_embedding table
        op.create_table(
            "chat_embedding",
            sa.Column("id", sa.Text(), nullable=False, primary_key=True),
            sa.Column("chat_id", sa.Text(), nullable=False),
            sa.Column("user_id", sa.Text(), nullable=False),
            sa.Column("role", sa.Text(), nullable=False),  # 'user' or 'assistant'
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("message_id", sa.Text(), nullable=True),  # Original message ID from chat JSON
            sa.Column("parent_id", sa.Text(), nullable=True),
            sa.Column("data", sa.JSON(), nullable=True),
            sa.Column("meta", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.Column("embedding", sa.Text(), nullable=True),
        )

        # Create indexes
        op.create_index("idx_chat_embedding_chat_id", "chat_embedding", ["chat_id"])
        op.create_index("idx_chat_embedding_user_id", "chat_embedding", ["user_id"])
        op.create_index("idx_chat_embedding_role", "chat_embedding", ["role"])
        op.create_index("idx_chat_embedding_created_at", "chat_embedding", ["created_at"])

        # Convert embedding to vector type (PostgreSQL only)
        op.execute("""
            ALTER TABLE chat_embedding 
            ALTER COLUMN embedding TYPE vector(1536) USING embedding::vector(1536)
        """)

        # Create vector index
        op.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_embedding_vector 
            ON chat_embedding USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
        """)


def downgrade() -> None:

    connection = op.get_bind()
    dialect = connection.dialect.name
    
    if dialect == 'postgresql':
    # Drop vector index
        op.execute("DROP INDEX IF EXISTS idx_chat_embedding_vector")
        
        # Drop regular indexes
        op.drop_index("idx_chat_embedding_created_at", "chat_embedding")
        op.drop_index("idx_chat_embedding_role", "chat_embedding")
        op.drop_index("idx_chat_embedding_user_id", "chat_embedding")
        op.drop_index("idx_chat_embedding_chat_id", "chat_embedding")
        
        # Drop table
        op.drop_table("chat_embedding")

"""update embedding table

Revision ID: 1a3a0ffb13e2
Revises: f75cfb55a211
Create Date: 2025-07-03 15:38:26.644219

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import open_webui.internal.db
from sqlalchemy.dialects import sqlite
from open_webui.migrations.util import get_existing_tables

# revision identifiers, used by Alembic.
revision: str = '1a3a0ffb13e2'
down_revision: Union[str, None] = 'f75cfb55a211'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Get database connection to check dialect
    connection = op.get_bind()
    dialect = connection.dialect.name
    
    if dialect == 'postgresql':
        # Safely drop old indexes if they exist (using IF EXISTS)
        op.execute("DROP INDEX IF EXISTS idx_chat_embedding_vector")
        op.execute("DROP INDEX IF EXISTS idx_chat_embedding_created_at")
        op.execute("DROP INDEX IF EXISTS idx_chat_embedding_role")
        op.execute("DROP INDEX IF EXISTS idx_chat_embedding_user_id")
        op.execute("DROP INDEX IF EXISTS idx_chat_embedding_chat_id")

        # Drop old table if it exists
        op.execute("DROP TABLE IF EXISTS chat_embedding")
        
        # Only available for postgresql with pgvector extension
        # connection = op.get_bind()
        # dialect = connection.dialect.name
        # if dialect != 'postgresql':
        #     pass

        existing_tables = set(get_existing_tables())

        if "chat_message" not in existing_tables:
            op.create_table(
                "chat_message",
                sa.Column("id", sa.String(255), nullable=False, primary_key=True), 
                sa.Column("chat_id", sa.String(255), nullable=False),  
                sa.Column("user_id", sa.String(255), nullable=False), 
                sa.Column("role", sa.String(255), nullable=False), 
                sa.Column("turn_number", sa.Integer(), nullable=False),
                sa.Column("content", sa.Text(), nullable=False),
                sa.Column("intent", sa.String(255), nullable=True),  
                sa.Column("topic", sa.String(255), nullable=True),  
                sa.Column("sentiment", sa.Float(), nullable=True),
                sa.Column("message_id", sa.String(255), nullable=True),  
                sa.Column("created_at", sa.BigInteger(), nullable=False),
                sa.Column("updated_at", sa.BigInteger(), nullable=False),
                sa.Column("embedding", sa.Text(), nullable=True),
            )
                   
            # Create indexes for optimal query performance
            # - most date-range queries
            # - top-intent charts
            # - show full convo ordered
            op.create_index("idx_chat_message_created_at_role", "chat_message", ["created_at", "role"])    
            op.create_index("idx_chat_message_intent_created_at", "chat_message", ["intent", "created_at"])
            op.create_index("idx_chat_message_chat_id_turn_number", "chat_message", ["chat_id", "turn_number"])
            
            # IVFFLAT on embedding (pgvector) – kNN search
            # - First convert embedding column to vector type
            # - Create IVFFLAT index for kNN search
            op.execute("""
                ALTER TABLE chat_message 
                ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384)
            """)
            
            op.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_message_embedding_vector 
                ON chat_message USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """)
            
            # Additional useful indexes
            op.create_index("idx_chat_message_user_id", "chat_message", ["user_id"])
            op.create_index("idx_chat_message_role", "chat_message", ["role"])
            op.create_index("idx_chat_message_created_at", "chat_message", ["created_at"])

        # Create message_chunk table for per-chunk embeddings
        if "chat_message_chunk" not in existing_tables:
            op.create_table(
                "chat_message_chunk",
                sa.Column("chunk_id", sa.String(255), nullable=False, primary_key=True),  
                sa.Column("msg_id", sa.String(255), nullable=False),  
                sa.Column("chunk_no", sa.SmallInteger(), nullable=False),
                sa.Column("embedding", sa.Text(), nullable=True),
            )
                
            # Create foreign key constraint
            op.create_foreign_key(
                "fk_chat_message_chunk_msg_id", 
                "chat_message_chunk", 
                "chat_message", 
                ["msg_id"], 
                ["id"], 
                ondelete="CASCADE"
            )
                
            # Create indexes for message_chunk table
            op.create_index("idx_chat_message_chunk_msg_id", "chat_message_chunk", ["msg_id"])
            op.create_index("idx_chat_message_chunk_chunk_no", "chat_message_chunk", ["chunk_no"])
            op.create_index("idx_chat_message_chunk_msg_id_chunk_no", "chat_message_chunk", ["msg_id", "chunk_no"])
                
            # Convert embedding column to vector type and create IVFFLAT index
            op.execute("""
                ALTER TABLE chat_message_chunk 
                ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384)
            """)
                
            op.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_message_chunk_embedding_vector 
                ON chat_message_chunk USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """)
        
        # ### end Alembic commands ###


def downgrade() -> None:
    # Drop message_chunk table and indexes first
    op.execute("DROP INDEX IF EXISTS idx_chat_message_chunk_embedding_vector")
    op.drop_index("idx_chat_message_chunk_msg_id_chunk_no", "message_chunk")
    op.drop_index("idx_chat_message_chunk_chunk_no", "chat_message_chunk")
    op.drop_index("idx_chat_message_chunk_msg_id", "chat_message_chunk")
    op.drop_table("chat_message_chunk")
    
    # Drop vector index first
    op.execute("DROP INDEX IF EXISTS idx_chat_message_embedding_vector")
    
    # Drop regular indexes
    op.drop_index("idx_chat_message_created_at_role", "chat_message")
    op.drop_index("idx_chat_message_intent_created_at", "chat_message")
    op.drop_index("idx_chat_message_chat_id_turn_number", "chat_message")
    op.drop_index("idx_chat_message_user_id", "chat_message")
    op.drop_index("idx_chat_message_role", "chat_message")
    op.drop_index("idx_chat_message_created_at", "chat_message")
    
    # Drop table
    op.drop_table("chat_message")
    # ### end Alembic commands ###

"""drop enrichment ai

Revision ID: 07332144a268
Revises: 36b5df363d24
Create Date: 2025-08-11 10:35:17.902198

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = '07332144a268'
down_revision: Union[str, None] = '36b5df363d24'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    bind = op.get_bind()
    dialect = bind.dialect.name
    insp = inspect(bind)
    idx_names = {i["name"] for i in insp.get_indexes("chat_message")}

    # Drop indexes that involve columns we're removing
    if "idx_chat_message_intent_created_at" in idx_names:
        op.drop_index("idx_chat_message_intent_created_at", table_name="chat_message")

    # pgvector index on embedding (Postgres only)
    if dialect == "postgresql":
        op.execute("DROP INDEX IF EXISTS idx_chat_message_embedding_vector")

    # Drop the enrichment columns
    cols_to_drop = ["intent", "topic", "sentiment", "embedding"]

    if dialect == "sqlite":
        with op.batch_alter_table("chat_message") as batch_op:
            for c in cols_to_drop:
                batch_op.drop_column(c)
    else:
        for c in cols_to_drop:
            op.drop_column("chat_message", c)


def downgrade():
    bind = op.get_bind()
    dialect = bind.dialect.name

    # Re-create columns
    add_cols = [
        sa.Column("intent", sa.String(length=255), nullable=True),
        sa.Column("topic", sa.String(length=255), nullable=True),
        sa.Column("sentiment", sa.Float(), nullable=True),
        sa.Column("embedding", sa.Text(), nullable=True),
    ]

    if dialect == "sqlite":
        with op.batch_alter_table("chat_message") as batch_op:
            for col in add_cols:
                batch_op.add_column(col)
    else:
        for col in add_cols:
            op.add_column("chat_message", col)

    # Re-create indexes
    op.create_index(
        "idx_chat_message_intent_created_at",
        "chat_message",
        ["intent", "created_at"],
    )

    # Restore pgvector index (Postgres only)
    if dialect == "postgresql":
        op.execute(
            """
            ALTER TABLE chat_message 
            ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384)
            """
        )
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_message_embedding_vector
            ON chat_message USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """
        )
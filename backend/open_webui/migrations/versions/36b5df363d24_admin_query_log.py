"""Admin_query_log

Revision ID: 36b5df363d24
Revises: 1a3a0ffb13e2
Create Date: 2025-07-17 15:27:15.008819

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import open_webui.internal.db


# revision identifiers, used by Alembic.
revision: str = '36b5df363d24'
down_revision: Union[str, None] = '1a3a0ffb13e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('admin_query_log',
        sa.Column('id', sa.String(255), nullable=False, primary_key=True),
        sa.Column('chat_id', sa.String(), nullable=False),
        sa.Column('user_msg_id', sa.String(255), nullable=True),
        sa.Column('ai_msg_id', sa.String(255), nullable=True),
        sa.Column('sql_text', sa.Text(), nullable=True),
        sa.Column('sql_summary', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('vector_summary', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.BigInteger(), server_default=sa.text('EXTRACT(EPOCH FROM NOW())::BIGINT'), nullable=False),
    )
    
    # Create indexes for performance
    op.create_index('ix_admin_query_log_chat_id', 'admin_query_log', ['chat_id'])
    op.create_index('ix_admin_query_log_created_at', 'admin_query_log', ['created_at'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_admin_query_log_created_at', table_name='admin_query_log')
    op.drop_index('ix_admin_query_log_chat_id', table_name='admin_query_log')
    
    # Drop table
    op.drop_table('admin_query_log')
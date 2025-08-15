"""add daily report table

Revision ID: 116f39494056
Revises: 07332144a268
Create Date: 2025-08-11 12:21:04.849430

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '116f39494056'
down_revision: Union[str, None] = '07332144a268'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('daily_report',
        sa.Column('id', sa.String(255), nullable=False, primary_key=True),
        sa.Column('chat_id', sa.String(255), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('conversation_analysis', sa.Text(), nullable=False),
        sa.Column('created_at', sa.BigInteger(), server_default=sa.text('EXTRACT(EPOCH FROM NOW())::BIGINT'), nullable=False),
    )
    
    # Create indexes for performance
    op.create_index('ix_daily_report_user_id', 'daily_report', ['user_id'])
    op.create_index('ix_daily_report_chat_id', 'daily_report', ['chat_id'])
    op.create_index('ix_daily_report_created_at', 'daily_report', ['created_at'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_daily_report_created_at', table_name='daily_report')
    op.drop_index('ix_daily_report_chat_id', table_name='daily_report')
    op.drop_index('ix_daily_report_user_id', table_name='daily_report')
    
    # Drop table
    op.drop_table('daily_report')
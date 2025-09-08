"""add report date

Revision ID: f18cf2e0fe0f
Revises: 116f39494056
Create Date: 2025-09-03 18:13:20.431556

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import open_webui.internal.db


# revision identifiers, used by Alembic.
revision: str = 'f18cf2e0fe0f'
down_revision: Union[str, None] = '116f39494056'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add report_date column to daily_report table
    op.add_column('daily_report', sa.Column('report_date', sa.BigInteger(), nullable=True))


def downgrade() -> None:
    # Remove report_date column from daily_report table
    op.drop_column('daily_report', 'report_date')

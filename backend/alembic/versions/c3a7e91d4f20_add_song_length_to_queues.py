"""add song_length to queues

Revision ID: c3a7e91d4f20
Revises: b8f4c2d1a903
Create Date: 2026-11-06 10:00:00.000000

Per-queue track-length mode. NULL = "full" (play tracks out); "half" caps
the OUT point around the midpoint so a set moves at DJ-slot pace.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c3a7e91d4f20'
down_revision: Union[str, None] = 'b8f4c2d1a903'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('queues', sa.Column('song_length', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('queues', 'song_length')

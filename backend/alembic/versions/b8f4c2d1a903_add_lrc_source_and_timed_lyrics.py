"""add lrc source + timed lyrics columns

Revision ID: b8f4c2d1a903
Revises: e9ea2bea0297
Create Date: 2026-11-04 12:20:00.000000

Adds what a timed `.lrc` sidecar gives us over a Genius scrape: where the
text came from, the line/word timestamps themselves, a script-derived
language hint for Whisper, and the fitted time warp used to correct the
drift hand-made `.lrc` files carry.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b8f4c2d1a903'
down_revision: Union[str, None] = 'e9ea2bea0297'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    lyricssource = sa.Enum('genius', 'lrc', 'embedded', name='lyricssource')
    lyricssource.create(op.get_bind(), checkfirst=True)

    op.add_column(
        'lyrics',
        sa.Column('source', lyricssource, nullable=False, server_default='genius'),
    )
    op.add_column(
        'lyrics',
        sa.Column(
            'line_timestamps',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        'lyrics', sa.Column('language', sa.String(), nullable=True)
    )
    op.add_column(
        'lyrics', sa.Column('alignment_offset', sa.Float(), nullable=True)
    )
    op.add_column(
        'lyrics', sa.Column('alignment_slope', sa.Float(), nullable=True)
    )
    # New column, so drop the default again to match the ORM (which
    # supplies it) rather than leaving a stale DB-side default behind.
    op.alter_column('lyrics', 'source', server_default=None)


def downgrade() -> None:
    op.drop_column('lyrics', 'alignment_slope')
    op.drop_column('lyrics', 'alignment_offset')
    op.drop_column('lyrics', 'language')
    op.drop_column('lyrics', 'line_timestamps')
    op.drop_column('lyrics', 'source')

    sa.Enum(name='lyricssource').drop(op.get_bind(), checkfirst=True)

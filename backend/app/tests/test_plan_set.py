"""Set-level planner helpers: capped-mode peak reporting."""

from __future__ import annotations

from app.workers.plan_set import _peak_energy_position


def test_peak_position_defaults_to_full_track():
    curve = [0.2] * 25 + [1.0] * 25 + [0.2] * 50
    assert _peak_energy_position(curve) == 0.25


def test_peak_position_ignores_a_late_peak_in_capped_mode():
    # Peak at ~60% of the full track; the capped half is all quiet.
    curve = [0.2] * 50 + [1.0] * 20 + [0.3] * 30
    assert _peak_energy_position(curve) > 0.5
    assert _peak_energy_position(curve, 0.5) == 0.0


def test_peak_position_keeps_an_early_peak_in_capped_mode():
    curve = [0.2] * 25 + [1.0] * 25 + [0.2] * 50
    # Normalised within the played half, so the same absolute peak reads
    # near the middle of that half rather than the middle of the track.
    assert 0.4 <= _peak_energy_position(curve, 0.5) <= 0.6

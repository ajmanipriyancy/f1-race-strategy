
import pytest
import numpy as np

from src.simulator import (
    simulate_strategy,
    generate_strategies,
    compare_strategies,
    pit_loss_sensitivity
)

class DummyModel:
    """A simple model that predicts zero lap-time degradation."""

    def predict(self, X):
        return np.zeros(len(X))


@pytest.fixture
def model():
    return DummyModel()


def test_valid_one_stop_simulation(model):
    summary, details = simulate_strategy(
        n_laps=20,
        pit_laps=[12],
        reference_pace=90,
        pit_loss=20,
        model=model,
        start_lap=7
    )

    assert summary["number_of_stops"] == 1
    assert summary["pit_loss_seconds"] == 20
    assert summary["estimated_remaining_time_seconds"] == 1280
    assert len(details) == 14


def test_pit_loss_calculation(model):
    summary, _ = simulate_strategy(
        n_laps=20,
        pit_laps=[12, 16],
        reference_pace=90,
        pit_loss=22,
        model=model,
        start_lap=7
    )

    assert summary["pit_loss_seconds"] == 44


def test_duplicate_pit_laps_are_rejected(model):
    with pytest.raises(ValueError, match="Duplicate pit laps"):
        simulate_strategy(
            n_laps=30,
            pit_laps=[12, 12],
            reference_pace=90,
            pit_loss=22,
            model=model,
            start_lap=7
        )


def test_invalid_pit_lap_is_rejected(model):
    with pytest.raises(ValueError, match="within the remaining race"):
        simulate_strategy(
            n_laps=30,
            pit_laps=[30],
            reference_pace=90,
            pit_loss=22,
            model=model,
            start_lap=7
        )


def test_negative_pit_loss_is_rejected(model):
    with pytest.raises(ValueError, match="cannot be negative"):
        simulate_strategy(
            n_laps=30,
            pit_laps=[15],
            reference_pace=90,
            pit_loss=-5,
            model=model,
            start_lap=7
        )


def test_long_stint_generates_reliability_warning(model):
    summary, _ = simulate_strategy(
        n_laps=35,
        pit_laps=[],
        reference_pace=90,
        pit_loss=22,
        model=model,
        start_lap=7
    )

    assert summary["reliability_warning"] == True
    assert summary["max_modelled_stint_lap"] == 35

def test_generate_one_stop_strategies():
    strategies = list(
        generate_strategies(
            n_laps=57,
            n_stops=1,
            min_stint=8,
            start_lap=7
        )
    )

    assert len(strategies) > 0

    for pits in strategies:
        stint_lengths = np.diff([6, *pits, 57])
        assert min(stint_lengths) >= 8


def test_generate_two_stop_strategies():
    strategies = list(
        generate_strategies(
            n_laps=57,
            n_stops=2,
            min_stint=8,
            start_lap=7
        )
    )

    assert len(strategies) > 0

    for pits in strategies:
        assert len(pits) == 2
        stint_lengths = np.diff([6, *pits, 57])
        assert min(stint_lengths) >= 8

def test_compare_strategies(model):
    comparison = compare_strategies(
        n_laps=57,
        reference_pace=90,
        pit_loss=22,
        model=model,
        start_lap=7,
        min_stint=8,
        max_stops=2
    )

    assert not comparison.empty
    assert list(comparison["strategy"]) == [
        "0-stop",
        "1-stop",
        "2-stop"
    ]

    assert comparison["estimated_remaining_time_seconds"].notna().all()


def test_pit_loss_sensitivity(model):
    sensitivity = pit_loss_sensitivity(
        n_laps=57,
        reference_pace=90,
        model=model,
        pit_losses=(0, 10, 20),
        start_lap=7,
        min_stint=8
    )

    assert len(sensitivity) == 3

    assert sensitivity["best_1_stop_time"].notna().all()
    assert sensitivity["best_2_stop_time"].notna().all()

    assert sensitivity["two_stop_advantage_seconds"].notna().all()

    # Increasing pit loss should not improve the estimated time
    assert sensitivity["best_1_stop_time"].is_monotonic_increasing
    assert sensitivity["best_2_stop_time"].is_monotonic_increasing


from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

PACE_FILE = PROCESSED / "pace_clean_candidate_2023_2024.csv"


def load_model():
    """
    Train the selected linear Ridge model on the eligible
    historical dataset, matching Notebook 05.
    """

    if not PACE_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {PACE_FILE}")

    pace = pd.read_csv(PACE_FILE)

    required = {
        "season", "round", "lap",
        "stint_lap", "delta",
    }

    missing = required - set(pace.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Construct race_id from season and round
    pace["race_id"] = (
    pace["season"].astype(str)
    + "_R"
    + pace["round"].astype(str)
    )

# Reproduce notebook race eligibility rule
    race_counts = pace.groupby("race_id").size()
    eligible_races = race_counts[race_counts >= 200].index

    data = pace[pace["race_id"].isin(eligible_races)].copy()

    X = data[["lap", "stint_lap"]]
    y = data["delta"]

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    )

    model.fit(X, y)

    return model, data


def simulate_strategy(
    n_laps,
    pit_laps,
    reference_pace,
    pit_loss,
    model,
    start_lap=7
):
    """
    Estimate remaining race time for a hypothetical strategy.

    Pit lap and following lap use reference-pace approximation.
    Other laps use the fitted linear Ridge model.

    pit_loss is an assumed total time loss per stop.
    """

    pit_laps = sorted(list(pit_laps))

    if len(pit_laps) != len(set(pit_laps)):
        raise ValueError("Duplicate pit laps are not allowed.")

    if any(p < start_lap or p >= n_laps for p in pit_laps):
        raise ValueError("Pit laps must be within the remaining race.")

    if pit_loss < 0:
        raise ValueError("Pit loss cannot be negative.")

    records = []

    for lap in range(start_lap, n_laps + 1):

        excluded_laps = set(pit_laps) | {
            p + 1 for p in pit_laps
        }

        if lap in excluded_laps:
            stint_lap = np.nan
            predicted_delta = 0.0
            estimated_lap_time = reference_pace

            if lap in pit_laps:
                lap_type = "Pit lap approximation"
            else:
                lap_type = "Post-pit lap approximation"

        else:
            previous_stops = [
                p for p in pit_laps if p < lap
            ]

            last_pit = max(previous_stops) if previous_stops else 0
            stint_lap = lap - last_pit

            features = pd.DataFrame({
                "lap": [lap],
                "stint_lap": [stint_lap]
            })

            predicted_delta = float(model.predict(features)[0])

            estimated_lap_time = (
                reference_pace + predicted_delta
            )

            lap_type = "Modelled lap"

        records.append({
            "lap": lap,
            "stint_lap": stint_lap,
            "lap_type": lap_type,
            "predicted_delta": predicted_delta,
            "estimated_lap_time": estimated_lap_time
        })

    lap_details = pd.DataFrame(records)

    pit_time_loss = len(pit_laps) * pit_loss

    estimated_time = (
        lap_details["estimated_lap_time"].sum()
        + pit_time_loss
    )

    max_stint = lap_details["stint_lap"].max()

    summary = {
        "pit_laps": pit_laps,
        "number_of_stops": len(pit_laps),
        "pit_loss_seconds": pit_time_loss,
        "estimated_remaining_time_seconds": estimated_time,
        "max_modelled_stint_lap": max_stint,
        "reliability_warning": (
            max_stint > 25 if pd.notna(max_stint) else False
        )
    }

    return summary, lap_details




def generate_strategies(
    n_laps,
    n_stops,
    min_stint=8,
    start_lap=7
):
    """Generate strategies satisfying minimum stint constraints."""

    possible_pits = range(
        start_lap + min_stint - 1,
        n_laps - min_stint + 1
    )

    for pits in itertools.combinations(possible_pits, n_stops):

        stint_lengths = np.diff(
            [start_lap - 1, *pits, n_laps]
        )

        if len(stint_lengths) and stint_lengths.min() >= min_stint:
            yield pits



def compare_strategies(
    n_laps,
    reference_pace,
    pit_loss,
    model,
    start_lap=7,
    min_stint=8,
    max_stops=2
):
    """Find the lowest estimated time for each stop count."""

    results = []

    for stops in range(max_stops + 1):

        candidates = generate_strategies(
            n_laps,
            stops,
            min_stint,
            start_lap
        )

        evaluated = []

        for pits in candidates:

            summary, details = simulate_strategy(
                n_laps=n_laps,
                pit_laps=pits,
                reference_pace=reference_pace,
                pit_loss=pit_loss,
                model=model,
                start_lap=start_lap
            )

            evaluated.append((summary, details))

        if not evaluated:
            continue

        best_summary, best_details = min(
            evaluated,
            key=lambda item: item[0][
                "estimated_remaining_time_seconds"
            ]
        )

        results.append({
            "strategy": f"{stops}-stop",
            "pit_laps": tuple(best_summary["pit_laps"]),
            "number_of_stops": stops,
            "estimated_remaining_time_seconds":
                best_summary["estimated_remaining_time_seconds"],
            "reliability_warning":
                best_summary["reliability_warning"]
        })

    return pd.DataFrame(results)


def pit_loss_sensitivity(
    n_laps,
    reference_pace,
    model,
    pit_losses=(0, 10, 15, 18, 22, 26, 30),
    start_lap=7,
    min_stint=8
):
    """Compare best one-stop and two-stop estimates across pit losses."""

    results = []

    for loss in pit_losses:

        best = {}

        for stops in (1, 2):

            candidates = []

            for pits in generate_strategies(
                n_laps, stops, min_stint,start_lap
            ):

                summary, _ = simulate_strategy(
                    n_laps=n_laps,
                    pit_laps=pits,
                    reference_pace=reference_pace,
                    pit_loss=float(loss),
                    model=model,
                    start_lap=start_lap
                )

                candidates.append((
                    summary["estimated_remaining_time_seconds"],
                    pits
                ))

            best[stops] = min(candidates)

        one_time, one_pits = best[1]
        two_time, two_pits = best[2]

        results.append({
            "pit_loss_per_stop": loss,
            "best_1_stop_lap": one_pits,
            "best_1_stop_time": one_time,
            "best_2_stop_laps": two_pits,
            "best_2_stop_time": two_time,
            "two_stop_advantage_seconds": one_time - two_time
        })

    return pd.DataFrame(results)

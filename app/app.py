
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator import (
    load_model,
    simulate_strategy,
    compare_strategies,
    pit_loss_sensitivity,
)

DATA_DIR = PROJECT_ROOT / "data" / "processed"

PREDICTIONS_FILE = (
    DATA_DIR / "heldout_race_lap_predictions_2023_2024.csv"
)

SUMMARY_FILE = DATA_DIR / "backtest_summary_2023_2024.csv"

st.set_page_config(
    page_title="F1 Race Strategy Intelligence",
    page_icon="🏎️",
    layout="wide",
)

st.title("🏎️ F1 Race Strategy Intelligence")
st.caption(
    "Historical lap-pace analysis and assumption-based "
    "pit-stop strategy simulation | 2023–2024"
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_predictions():
    return pd.read_csv(PREDICTIONS_FILE)


@st.cache_data
def get_summary():
    return pd.read_csv(SUMMARY_FILE)


# Load data
try:
    model, training_data = get_model()
    predictions = get_predictions()
    summary = get_summary()

except Exception as error:
    st.error(f"Could not load project data: {error}")
    st.stop()


# Backtest metrics

# Backtest metrics
st.header("Model Backtesting")

overall = summary[
    summary["evaluation"] == "Overall held-out race"
].iloc[0]

baseline_mae = float(overall["baseline_MAE"])
ridge_mae = float(overall["ridge_MAE"])

baseline_rmse = float(overall["baseline_RMSE"])
ridge_rmse = float(overall["ridge_RMSE"])

improvement = float(overall["MAE_improvement_pct"])

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Training races",
    len(training_data["race_id"].unique())
)

col2.metric(
    "Out-of-fold predictions",
    f"{len(predictions):,}"
)

col3.metric(
    "Baseline MAE",
    f"{baseline_mae:.3f} s"
)

col4.metric(
    "Ridge MAE",
    f"{ridge_mae:.3f} s",
    delta=f"{improvement:.2f}% improvement"
)

st.caption(
    f"Baseline RMSE: {baseline_rmse:.3f} s | "
    f"Ridge RMSE: {ridge_rmse:.3f} s"
)

st.info(
    "Ridge improves lap-level MAE by 12.15% across 35 held-out "
    "races. However, total-level backtesting showed worse "
    "aggregate error than the baseline. This is a lap-pace "
    "model, not a full-race finishing-time predictor."
)





# Historical analysis
st.header("Historical Lap-Pace Analysis")

required_columns = {
    "season",
    "round",
    "driver_id",
    "lap",
    "delta",
    "ridge_predicted_delta",
    "lap_time_seconds",
    "ridge_predicted_lap_time",
}

missing = required_columns - set(predictions.columns)

if missing:
    st.error(f"Prediction file is missing columns: {missing}")
    st.stop()

race_options = (
    predictions[["season", "round"]]
    .drop_duplicates()
    .sort_values(["season", "round"])
)

race_labels = {
    f"{int(row.season)} - Round {int(row.round)}":
    (int(row.season), int(row.round))
    for row in race_options.itertuples()
}

selected_race = st.selectbox(
    "Select race",
    list(race_labels.keys()),
    key="historical_race_select",
)

season, round_number = race_labels[selected_race]

race_data = predictions[
    (predictions["season"] == season)
    & (predictions["round"] == round_number)
].copy()

drivers = sorted(race_data["driver_id"].unique())

selected_driver = st.selectbox(
    "Select driver",
    drivers,
    key="historical_driver_select",
)

driver_data = race_data[
    race_data["driver_id"] == selected_driver
].sort_values("lap")

if not driver_data.empty:

    st.subheader("Lap-time delta: Actual vs Ridge")

    delta_chart = driver_data.set_index("lap")[
        ["delta", "ridge_predicted_delta"]
    ].rename(columns={
        "delta": "Actual delta",
        "ridge_predicted_delta": "Ridge predicted delta",
    })

    st.line_chart(delta_chart)

    st.subheader("Lap time: Actual vs Ridge")

    time_chart = driver_data.set_index("lap")[
        ["lap_time_seconds", "ridge_predicted_lap_time"]
    ].rename(columns={
        "lap_time_seconds": "Actual lap time",
        "ridge_predicted_lap_time": "Ridge predicted lap time",
    })

    st.line_chart(time_chart)

    st.caption(
        "These are out-of-fold predictions: each race was "
        "held out while its model predictions were generated."
    )

    st.dataframe(
        driver_data[
            [
                "lap",
                "lap_time_seconds",
                "ridge_predicted_lap_time",
                "delta",
                "ridge_predicted_delta",
                "ridge_error",
            ]
        ],
        use_container_width=True,
    )





# Strategy simulator
st.header("Pit-Stop Strategy Simulator")

st.warning(
    "This is a hypothetical simulator. Pit loss is an assumed "
    "total time penalty, not the API pit-lane duration. Pit laps "
    "and the following lap are approximated at reference pace. "
    "Tyre compound, fuel, weather, traffic and safety-car data "
    "are not modelled."
)

with st.form("strategy_form"):

    col1, col2, col3 = st.columns(3)

    race_laps = col1.number_input(
        "Total race laps",
        min_value=20,
        max_value=100,
        value=57,
    )

    start_lap = col2.number_input(
        "Start simulation from lap",
        min_value=1,
        max_value=99,
        value=7,
    )

    reference_pace = col3.number_input(
        "Reference lap pace (seconds)",
        min_value=50.0,
        max_value=150.0,
        value=97.9954,
        step=0.1,
    )

    col4, col5 = st.columns(2)

    pit_loss = col4.number_input(
        "Assumed total pit loss per stop (seconds)",
        min_value=0.0,
        max_value=60.0,
        value=22.0,
        step=1.0,
    )

    pit_laps_input = col5.text_input(
        "Pit laps, comma-separated",
        value="14,36",
    )

    submitted = st.form_submit_button(
        "Simulate strategy",
        type="primary",
    )


if submitted:

    try:
        pit_laps = [
            int(value.strip())
            for value in pit_laps_input.split(",")
            if value.strip()
        ]

        result, lap_details = simulate_strategy(
            n_laps=int(race_laps),
            pit_laps=pit_laps,
            reference_pace=reference_pace,
            pit_loss=pit_loss,
            model=model,
            start_lap=int(start_lap),
        )

        st.subheader("Simulation Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Estimated remaining time",
            f"{result['estimated_remaining_time_seconds']:.2f} s",
        )

        col2.metric(
            "Number of stops",
            result["number_of_stops"],
        )

        col3.metric(
            "Assumed total pit loss",
            f"{result['pit_loss_seconds']:.1f} s",
        )

        if result["reliability_warning"]:
            st.warning(
                "At least one modelled stint exceeds 25 laps. "
                "Predictions are outside the model's reliable range."
            )
        else:
            st.success(
                "All modelled stints are within the "
                "backtested range of up to 25 laps."
            )

        st.line_chart(
            lap_details.set_index("lap")[
                ["estimated_lap_time"]
            ]
        )

        st.dataframe(
            lap_details,
            use_container_width=True,
        )

    except Exception as error:
        st.error(f"Simulation error: {error}")




# Strategy Comparison
st.header("Strategy Comparison")

st.caption(
    "Compares the fastest estimated strategy for each number "
    "of pit stops under the selected assumptions."
)

if submitted:
    try:
        comparison = compare_strategies(
            n_laps=int(race_laps),
            reference_pace=reference_pace,
            pit_loss=pit_loss,
            model=model,
            start_lap=int(start_lap),
            min_stint=8,
            max_stops=2
        )

        if comparison.empty:
            st.warning("No valid strategies could be generated.")

        else:
            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Estimated Time by Stop Count")

            chart_data = comparison[
                ["strategy", "estimated_remaining_time_seconds"]
            ].set_index("strategy")

            st.bar_chart(chart_data)

            if comparison["reliability_warning"].any():
                st.warning(
                    "Some strategies contain modelled stints "
                    "outside the backtested reliability range."
                )
            else:
                st.success(
                    "All compared strategies are within the "
                    "backtested stint-length range."
                )

    except Exception as error:
        st.error(f"Strategy comparison error: {error}")



# Pit-Loss Sensitivity Analysis
st.header("Pit-Loss Sensitivity Analysis")

st.caption(
    "Shows how the estimated best one-stop and two-stop "
    "strategies change as assumed pit loss varies."
)

if submitted:
    try:
        # Calculate sensitivity results
        sensitivity = pit_loss_sensitivity(
            n_laps=int(race_laps),
            reference_pace=reference_pace,
            model=model,
            start_lap=int(start_lap),
            min_stint=8
        )

        # Display results table
        st.subheader("Sensitivity Results")

        st.dataframe(
            sensitivity,
            use_container_width=True,
            hide_index=True
        )

        # Prepare data for plotting
        sensitivity_plot = sensitivity.copy()

        baseline_1_stop = sensitivity_plot[
            "best_1_stop_time"
        ].iloc[0]

        baseline_2_stop = sensitivity_plot[
            "best_2_stop_time"
        ].iloc[0]

        sensitivity_plot["1_stop_time_increase"] = (
            sensitivity_plot["best_1_stop_time"]
            - baseline_1_stop
        )

        sensitivity_plot["2_stop_time_increase"] = (
            sensitivity_plot["best_2_stop_time"]
            - baseline_2_stop
        )

        # Chart 1: Impact of pit loss
        st.subheader("Impact of Pit Loss on Strategy Time")

        st.caption(
            "Shows the additional estimated time compared "
            "with the zero-pit-loss scenario."
        )

        st.line_chart(
            sensitivity_plot,
            x="pit_loss_per_stop",
            y=[
                "1_stop_time_increase",
                "2_stop_time_increase"
            ]
        )

        # Chart 2: Two-stop advantage
        st.subheader("Two-Stop Advantage")

        st.caption(
            "Positive values mean two stops are faster. "
            "Negative values mean one stop is faster."
        )

        st.line_chart(
            sensitivity_plot,
            x="pit_loss_per_stop",
            y="two_stop_advantage_seconds"
        )

    except Exception as error:
        st.error(
            f"Sensitivity analysis error: {error}"
        )

st.divider()

st.caption(
    "Educational analytics project. Estimated times are "
    "model outputs under stated assumptions, not live race "
    "predictions or verified optimal strategies."
)

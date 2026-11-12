# F1 Race Strategy Intelligence

An end-to-end Formula 1 analytics project exploring race pace, pit-stop strategy, and predictive modelling using historical F1 data.

## Overview

F1 Race Strategy Intelligence analyses historical Formula 1 lap and pit-stop data to understand race pace behaviour and evaluate hypothetical pit-stop strategies.

The project combines data collection, preprocessing, exploratory data analysis, feature engineering, machine learning, race-wise validation, and an interactive Streamlit dashboard.

The objective is to investigate whether historical lap-level patterns can help estimate pace and compare hypothetical race strategies.

## Project Highlights

- Analysed 46 races across the 2023 and 2024 seasons.
- Processed approximately 50,960 lap records and 1,737 pit-stop records.
- Built an eligible modelling dataset containing 29,901 observations across 35 races.
- Developed a Ridge Regression model using lap number and stint lap as features.
- Used race-wise out-of-fold validation to evaluate generalisation to unseen races.
- Improved lap-level MAE from 1.0803 seconds to 0.9491 seconds, approximately 12.15%.
- Ridge outperformed the baseline in 31 of 35 evaluated races.
- Built a Streamlit dashboard with strategy simulation, strategy comparison, and pit-loss sensitivity analysis.
- Added automated tests for simulator calculations and input validation.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Jupyter Notebook
- Pytest
- Git and GitHub

## Project Structure

```text
f1-project/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_race_data_exploration.ipynb
│   ├── 02_pit_stop_data.ipynb
│   ├── 03_cleaning_and_target.ipynb
│   ├── 04_data_expansion_and_model_interpretation.ipynb
│   └── 05_pit_stop_scenario_simulator.ipynb
├── src/
│   ├── __init__.py
│   └── simulator.py
├── tests/
│   └── test_simulator.py
├── DECISIONS.md
├── requirements.txt
└── README.md
```

## Methodology

### 1. Data Collection and Exploration

Historical Formula 1 race, lap, and pit-stop data from 2023 and 2024 were collected and organised for analysis.

The initial workflow involved examining race-level data, lap records, pit-stop events, and data quality.

### 2. Data Cleaning and Feature Engineering

The data preprocessing workflow included filtering unsuitable observations, handling pit-stop-related records, constructing stint-level features, and preparing the modelling target.

The final eligible dataset contained 29,901 observations across 35 races.

The model uses:
- `lap`: Current race lap.
- `stint_lap`: Lap number within the current tyre stint.
- `delta`: Lap-time difference relative to the selected reference pace.

### 3. Predictive Modelling

A Ridge Regression model was implemented using a Scikit-learn pipeline containing:

- StandardScaler
- Ridge Regression with alpha = 1.0

The model uses lap number and stint lap to estimate lap-time deviation from reference pace.

### 4. Model Validation

Race-wise out-of-fold validation was used to evaluate the model on held-out races rather than relying only on random row-level splitting.

| Metric | Baseline | Ridge Regression |
|---|---:|---:|
| Lap-level MAE | 1.0803 s | 0.9491 s |

The Ridge model reduced lap-level MAE by approximately 12.15% and outperformed the baseline in 31 out of 35 evaluated races.

**Important limitation:** The improvement in lap-level MAE does not establish improved total race-time prediction. The aggregate race-time backtest was worse for Ridge than for the baseline. The model should therefore be interpreted as a lap-pace estimation model, not a complete race-outcome predictor.

## Strategy Simulator

The simulator estimates remaining race time under hypothetical pit-stop plans.

It supports:

- User-defined pit laps.
- Configurable reference pace.
- Configurable pit-loss assumptions.
- One-stop and multi-stop strategy comparisons.
- Pit-loss sensitivity analysis.
- Reliability warnings for modelled stints exceeding 25 laps.

The dashboard also compares estimated times across different numbers of stops and examines how changing pit-loss assumptions affects strategy selection.

## Simulator Assumptions and Limitations

The simulator is an analytical approximation, not a full Formula 1 race simulation.

Current assumptions include:

- Pit laps and the following lap are approximated using reference pace.
- Pit loss is treated as a fixed total time penalty per stop.
- Tyre compound differences are not explicitly modelled.
- Fuel load, weather, traffic, safety cars, and virtual safety cars are not modelled.
- Modelled stints exceeding 25 laps are flagged as outside the more reliable backtested range.
- A mathematically fastest no-stop strategy is not necessarily legal under actual F1 tyre regulations.

These limitations should be considered when interpreting the results.

## Interactive Dashboard

The Streamlit dashboard provides:

1. Historical lap-pace analysis.
2. Hypothetical pit-stop strategy simulation.
3. Strategy comparison across different stop counts.
4. Pit-loss sensitivity analysis.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/ajmanipriyancy/f1-race-strategy.git
cd f1-race-strategy
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app/app.py
```

The processed dataset required by the application must be available at:

```text
data/processed/pace_clean_candidate_2023_2024.csv
```

### 5. Run automated tests

```bash
python -m pytest -v
```

## Testing

The simulator has an automated pytest suite covering:

- Valid one-stop simulations.
- Pit-loss calculations.
- Duplicate pit-lap rejection.
- Invalid pit-lap rejection.
- Negative pit-loss rejection.
- Long-stint reliability warnings.
- One-stop and two-stop strategy generation.
- Strategy comparison.
- Pit-loss sensitivity analysis.

Current result: **10 tests passed.**

## Future Improvements

Potential extensions include:

- Tyre compound and degradation modelling.
- Fuel-load estimation.
- Weather and track-condition integration.
- Safety-car and traffic effects.
- More advanced predictive models.
- More realistic race-level strategy optimisation.

## Author

Priyancy Ajmani  
B.E. Computer Engineering  
Thapar Institute of Engineering and Technology

GitHub: https://github.com/ajmanipriyancy
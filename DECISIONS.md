# Decisions

1. Pit "duration" has a median of ~23 s, so it is time spent in the pit lane (entry to exit), not
   stationary time. It is not the same as total time lost in a stop.
2. 67 of the 70 missing pit durations are in two races (2023 R3: 49, 2024 R4: 18), both of which
   had race-wide slow laps that look like red flags. I kept the missing values as blanks because
   filling them in would mean inventing numbers.
3. A lap is flagged "odd" if its time is above 1.07x or below 0.90x the race's median lap time.
   This is a heuristic: I have no safety-car or red-flag data, so I can only guess from lap times.
4. Target = lap time minus that driver's median clean lap in that race ("delta"). Circuits have
   different lap lengths, so raw lap times cannot generalise to a race the model has not seen.
   Delta isolates the shape (fuel burn-off, stint effects) from the pace level. The pace level is
   given to the model, not predicted.
5. I excluded lap 1, the pit lap, and the lap after the pit lap. Median delta was +7.5 s on the pit
   lap and +22.0 s on the following lap, compared with +0.15 s two laps after.
6. stint_lap counts laps since the driver's last recorded stop. It is a proxy for tyre age, but I
   do not have compound data, so it is only an approximation.
7. Evaluation is leave-one-race-out (train on 9 races, test on the 10th), not a random split of
   laps, because laps from the same race are near-duplicates and would leak between train and test.

## Day 4 — Initial Race-Pace Modelling

### Dataset
- 6,342 prediction rows
- 8 races across 2023 and 2024
- 156 driver-race groups
- Reference pace calculated from each driver's first 5 eligible laps
- Target: lap-time deviation from reference pace

### Models evaluated

| Model | MAE (s) | RMSE (s) |
|---|---:|---:|
| Zero-delta baseline | 1.5226 | 1.7919 |
| Training-mean baseline | 0.8828 | 1.1514 |
| Ridge (alpha=1) | 0.6640 | 0.9197 |
| Nested tuned Ridge | 0.6624 | 0.9190 |
| Wider alpha Ridge | 0.6671 | 0.9240 |
| Random Forest | 0.6685 | 0.9273 |

### Decision
- Retain Ridge (alpha=1) as the current benchmark.
- Nested tuning produced negligible improvement.
- Random Forest did not outperform Ridge.
- Prioritise feature engineering before further model tuning.

### Limitations
- Only 8 races evaluated.
- Stint lap is a proxy, not verified tyre age.
- Retrospective anomaly filtering was applied.
- Current model predicts clean-lap pace deviation, not race outcomes or optimal pit strategy.

## Day 5 — Expanded data and model validation (2023–2024)

### Data collection and quality

- Collected lap and pit-stop records across 46 races from the 2023 and 2024 seasons.
- Lap dataset: 50,960 rows; zero duplicate race-driver-lap keys.
- Pit-stop dataset: 1,737 records; zero duplicate race-driver-stop keys.
- One pit-stop duration remained missing and was preserved as NaN.
- Reconciled all 374 pit stops from the original first-10-race dataset against the expanded dataset.
- The 70 previously missing durations were parsing failures for m:ss.mmm values. All 374 matched records agreed after correct parsing.

### Target construction

- Converted lap times to seconds.
- Excluded lap 1, pit laps, and the lap immediately following a pit stop.
- Calculated each driver's reference pace from their first five eligible laps.
- Target: delta = lap_time_seconds - reference_pace.
- Created stint_lap from recorded pit-stop laps. This is a pit-stop-derived proxy, not verified tyre age.

### Retrospective filtering

- Initial prediction rows: 42,323 across 46 races.
- Explicitly excluded 2023 R3 and 2024 R4 based on the earlier distortion analysis.
- Excluded race-lap groups with at least five drivers and absolute median delta greater than 5 seconds.
- This removed 10,590 rows and eliminated six additional races entirely: 2023 R16, R17, R21 and 2024 R19, R20, R24.
- Remaining rows after race-lap filtering: 30,258 across 38 races.
- Additionally excluded 204 individual observations with absolute delta greater than 10 seconds (0.67%).
- Final clean-pace candidate: 30,054 rows across 38 races.
- Preserved both the all-filtered and clean-candidate datasets.

These exclusions use completed-race information and are retrospective. They define a normal-pace research subset, not an unbiased prospective sample of all race conditions.

### Model comparison

Features: lap and stint_lap.

Validation: Leave-One-Race-Out (LORO), with race as the held-out group.

| Dataset | Model | MAE (s) | RMSE (s) |
|---|---|---:|---:|
| All filtered | Zero baseline | 1.6592 | 2.7137 |
| All filtered | Training-mean baseline | 1.2068 | 2.3999 |
| All filtered | Ridge | 1.0728 | 2.3460 |
| All filtered | Random Forest | 1.1461 | 2.4236 |
| Clean candidate | Zero baseline | 1.5430 | 1.9721 |
| Clean candidate | Training-mean baseline | 1.0893 | 1.4910 |
| Clean candidate | Ridge | 0.9592 | 1.4194 |
| Clean candidate | Random Forest | 0.9979 | 1.4928 |

Ridge was the best-performing model among those tested on both datasets.

### Robust evaluation

- Required at least 200 prediction rows per race.
- Retained 35 races and 29,901 rows.
- Training-mean baseline: MAE 1.0803 s, RMSE 1.4721 s.
- Ridge: MAE 0.9491 s, RMSE 1.3983 s.
- Ridge reduced pooled MAE by approximately 12.1% and RMSE by approximately 5.0%.
- Ridge beat the training-mean baseline in 31 of 35 held-out races (88.6%).
- Median race-wise Ridge MAE: 0.8371 s.
- Ridge performed worse than the baseline in 2023 R19, 2023 R6, 2024 R11 and 2024 R8.

### Limitations and interpretation

- The model was evaluated on 38 retained races, not all 46 collected races.
- The 35-race robust subset is selected retrospectively by row count.
- Filtering based on completed-race target values makes the clean-pace results optimistic for prospective deployment.
- Only two model features were tested: lap and stint_lap.
- stint_lap is not verified tyre age.
- Model coefficients and predictions are associations within this dataset; they do not directly measure fuel burn or tyre degradation.
- Jolpica data does not provide verified tyre compounds, tyre age, weather, fuel load, or total pit-lane loss.
- These results establish a clean-pace baseline, not a complete race-strategy optimizer.

### Day 5 decision

Use Ridge Regression with lap and stint_lap as the first baseline model for the next project stage. Keep Random Forest as a comparison model. Preserve the filtering limitations and race-level failures in future reporting.



## Day 6 — Pit-stop scenario simulator

- Built a hypothetical remaining-race simulator using the linear Ridge clean-pace model.
- Selected linear Ridge for transparency; the quadratic model's MAE advantage was only about 0.0086 seconds.
- Ordinary laps use race lap and stint-lap features. A pit lap and the following lap are approximated at reference pace.
- The API's pit duration (median approximately 23 seconds) represents time in the pit lane from entry to exit, not the 2–3 seconds the car is stationary. It is not total time lost, because the car would have covered that stretch of track anyway. The simulator uses an assumed total pit loss, varied in sensitivity analysis.
- Demonstrated on 2023 Bahrain GP, Max Verstappen, using an assumed 22-second pit loss.
- Compared no-stop, one-stop, two-stop and historical-stop scenarios; tested sensitivity to pit loss.
- No-stop is a reference only, not a legal dry-race strategy under the two-compound rule.
- The simulator does not model compounds, physical tyre degradation, fuel, traffic, weather or safety cars.
- The Bahrain demonstration model was trained on all 35 eligible races, including Bahrain. This is an in-sample demonstration.



## Day 7 — Historical backtesting

- Generated out-of-fold predictions by holding out one complete race at a time and training on the other 34 races.
- Evaluated 35 eligible races and 29,901 clean prediction rows.
- Training-mean baseline: MAE 1.0803 seconds; RMSE 1.4721 seconds.
- Linear Ridge: MAE 0.9491 seconds; RMSE 1.3983 seconds.
- Ridge reduced pooled MAE by 12.15% and beat the baseline in 31 of 35 races (88.6%).
- Mean race-level MAE: 1.1578 seconds for baseline and 1.0294 seconds for Ridge.
- Ridge performed better in stint groups 2–25; the baseline performed better in groups 26–58.
- Long-stint results are a limitation. The model lacks tyre compound, verified tyre age, fuel, traffic, weather and safety-car information.
- Reference pace is derived retrospectively from each driver-race's first five eligible laps. This is not a live race-start forecast.
- These results validate clean-pace prediction on the selected historical dataset; they do not yet validate the strategy simulator's race-time accuracy.


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
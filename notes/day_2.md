Collected pit-stop data for the same 10 races as the lap dataset.

Flattened API records into a consistent DataFrame.

Validated missing values and duplicate stop keys.

Found 70 missing durations, concentrated mainly in 2023 Round 3 and 2024 Round 4.

Saved pit_stops_2023_2024_first5.csv.

Joined pit-stop lap keys to lap data using season, round, driver, and lap.

Verified the merge preserved 10,050 rows and created 374 pit-lap flags.

Saved laps_with_pit_flags_2023_2024_first5.csv.

Limitation: pit-stop duration is not total pit-lane time loss; pit-lap flags do not identify clean laps by themselves.
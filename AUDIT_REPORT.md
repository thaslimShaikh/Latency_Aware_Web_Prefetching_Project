# NEXUS technical audit report

Audit date: 2026-10-18. Environment: Python 3.12.3, scikit-learn 1.6.1 (as pinned), Streamlit 1.66.0, pandas 3.0.6, numpy 2.5.4.
Original project preserved untouched in `nexus_ORIGINAL_BACKUP/`. `models/` was **not** modified.

## 0. What could and could not be verified

| Item | Status |
|---|---|
| Source code, saved model, metadata, tests | Inspected in full |
| Training notebook, feature-generation scripts, datasets | **Not in the zip.** Nothing about training was read from code; training facts come only from the comments in `forecast.py`, `global_model_summary.json` and `LIMITATIONS.md` |
| Held-out RMSE / MAE / R² | **Not recomputed.** They are values recorded from the notebook. I checked them for internal consistency (§3) but could not reproduce them |
| Lag/timestamp semantics vs training | **Not independently verified** (needs the CSVs: `tools/verify_against_training_csv.py`) |
| Seasonal-naive baseline | **Not evaluated** (no data, and a verified sampling interval is a precondition) |
| Candidate model | **Not produced** (no training data) |

## 1. One prediction traced end to end

| # | Question | Finding |
|---|---|---|
| 1 | Parsing and order | `parse_pasted` splits on commas/semicolons/whitespace and converts to float; order is *assumed* oldest-first and cannot be checked for pasted values (a reversed paste silently changes every lag). CSV rows are sorted by timestamp; duplicates, bad timestamps and non-numeric values are rejected. Only the last 60 values are used (a note is shown if more are supplied) |
| 2 | Lags | `lag_k = readings[-k]` for k in 1, 2, 3, 5, 10, 15, 30, 60; `readings[-1]` (newest) is `lag_1` of the target one step ahead; `lag_60` is the oldest of the 60. Unit-tested (`test_lag_k_is_k_steps_before_target_and_order_matters`) |
| 3 | What a lag means | One native sampling step: 60 min for IT power, site power, PUE; 1 min for circuits 1–3 and CRAC/ULC units. This matches the UI labels. **Not verified against training code.** Indirect support only: row counts (e.g. NREL 1,747 test rows ≈ 20 % of ~8.7 k hourly rows ≈ one year; Danish 4,560 test rows ≈ 20 % of ~22.8 k per-minute rows ≈ 16 days) |
| 4 | Time features | Target time = last timestamp + 1 step, UTC. `minute = hour*60+minute`, day-of-week (Mon = 0), hour, month−1, each as sin/cos. Formula is unit-tested; equality with training features is **unverified** |
| 5 | Signal routing | `Signal.key → series_id →` one of 10 one-hot `series_*` columns; scaling, training range and metrics are all looked up by the same `series_id`. Tested for uniqueness and one-hot correctness |
| 6 | Invalid input | Non-finite, <60 readings, negative power, PUE outside 1–2, readings > 1000 × training max (new, unit-error guard) are rejected with messages. Outside the training range → warning. New: readings in the model's top input bin → warning |
| 7 | Inverse transform | `prediction = model_output × scale + center`, per series, `center`/`scale` = median/IQR of the training split. Tested for all 10 signals |
| 8 | Displayed value | The inverse-transformed value is shown unchanged in the signal's own unit; delta and % are relative to the last reading |
| 9 | Errors / reliability | RMSE, MAE, R² and baseline values are **static** numbers from `global_model_summary.json` (not computed live). Tier comes from `assess_reliability` (§4) |

## 2. Root cause of the forecast problem (IT power: RMSE 738 kW vs baseline 67 kW)

**Not caused by (evidence):**
- Inverse scaling / routing / feature order: model checksum matches metadata; 26 feature names and order are verified at load; formula and routing are unit-tested; the reported 759.9 kW reproduces exactly (steady 730 kW input, hour-of-day sweep gives 751.2–759.9 kW).
- Metrics from mixed runs or misaligned rows: from RMSE and R² one can back out the test-set standard deviation. For all 10 series the value implied by the model row and by the baseline row agree to within 0.03 % (`tools/audit_model_probe.py`, Check A), so both were scored on the same test rows, in original units.

**Established cause (reproducible from the saved model alone, Probe B):** the model cannot represent “stay at the current level” above about 1,947 kW. The highest bin threshold of `lag_1` is 1,947.3; every IT-power input above it falls in one bin and produces the same forecast, **3,288.9 kW**:

| constant input | 1,500 | 1,900 | 2,000 | 2,600 | 4,000 |
|---|---|---|---|---|---|
| forecast (kW) | 1,447 | 1,930 | **3,289** | **3,289** | **3,289** |

Below 1,900 kW the model tracks the input to within roughly ±55 kW on this synthetic probe. `LIMITATIONS.md` (from the notebook, not re-verified) states that the test period median is ≈ 2,656 kW, above that threshold. A forecast pinned at 3,289 kW against actuals near 2,656 kW gives errors of several hundred kW, consistent with RMSE 738 kW. This is an inference linking a verified model behaviour to an unverified test-set description.

**Contributing design choices (verified in code/metadata):**
- Lags are fed as raw values while the target is scaled per series, and one tree model is shared by 10 series whose raw lag values overlap in range. Histogram binning (255 bins) is computed over the pooled values, so resolution per series is coarse and no bin above 1,947 exists (e.g. only 5 `lag_1` thresholds fall inside the whole NREL site-power training range).
- The model was fitted on train + validation (80 %), but the stored “training range” and scaling come from the first 60 %. The 3,289 kW output can only come from fit-data targets above the recorded training maximum of 2,216 kW, so the range used for warnings under-describes what the model saw (inferred).

**Not established:** why PUE, circuit 3 and CRAC-3/4 are worse than baseline (needs the data). For circuit 3 the training scale (849.6 W) is about 200× the test-set standard deviation (4.4 W), which may be a regime change, but this is a hypothesis.

**Is retraining justified?** The evidence points that way (a residual target `(y − lag_1)/scale`, or scale-normalised lags, would make persistence the default). I did not retrain: the data is not available and nothing may be claimed without evaluation. Any candidate must go in a separate folder and be compared per signal against persistence on the same chronological test split before replacing the current artifact.

## 3. Recorded held-out performance per signal (original units; NOT recomputed)

“Δ” = model vs last-reading baseline; + means the model is **worse**. Evaluation period is not stored in the metadata.

| Signal | Unit | n_test | RMSE model / base | ΔRMSE | MAE model / base | ΔMAE | R² model / base | Dashboard tier |
|---|---|---|---|---|---|---|---|---|
| IT power | kW | 12,112 | 737.9 / 67.25 | +997 % | 608 / 28.89 | +2004 % | −0.88 / 0.98 | Low |
| Site power | kW | 1,747 | 4.286 / 5.115 | −16 % | 2.967 / 3.089 | −4 % | −0.66 / −1.37 | Limited |
| PUE | ratio | 15,894 | 0.03738 / 0.00653 | +472 % | 0.01472 / 0.00144 | +922 % | 0.39 / 0.98 | Low |
| Circuit 1 | W | 4,560 | 19.73 / 42.89 | −54 % | 15.53 / 38.17 | −59 % | 0.55 / −1.15 | Moderate |
| Circuit 2 | W | 4,560 | 4.543 / 4.817 | −6 % | 2.872 / 2.311 | **+24 %** | 0.66 / 0.62 | **Limited** (was Moderate) |
| Circuit 3 | W | 4,560 | 20.93 / 4.176 | +401 % | 17.10 / 1.584 | +980 % | −21.99 / 0.09 | Low |
| CRAC-3 | kW* | 11,814 | 2.889 / 2.643 | +9 % | 1.664 / 0.958 | +74 % | 0.95 / 0.96 | Low |
| CRAC-4 | kW* | 11,817 | 3.996 / 2.221 | +80 % | 2.172 / 0.725 | +200 % | 0.89 / 0.97 | Low |
| ULC-5 | kW* | 11,732 | 0.1060 / 0.1017 | +4 % | 0.02777 / 0.02977 | −7 % | 0.06 / 0.14 | Low |
| ULC-6 | kW* | 11,793 | 0.03081 / 0.03414 | −10 % | 0.02353 / 0.02499 | −6 % | 0.70 / 0.63 | Moderate |

\*Unit assumed; the notebook does not state it. Metrics are in each signal's own unit and must not be pooled across signals.
**Conclusion: the saved model is not suitable for operational forecasting on 6 of 10 signals (RMSE worse than repeating the last reading), and only 3 signals beat the baseline on both RMSE and MAE.** Reproduce this table with `python tools/audit_model_probe.py`.

## 4. Reliability display (changed)

Four states are now distinguishable:
1. **Not generated** – input/model validation failed: red “Forecast not produced” message, no number shown.
2. **Generated, worse than baseline (Low)** – amber banner; last reading named as the operational reference; baseline RMSE, model RMSE, MAE and R² shown.
3. **Generated, mixed / limited (Limited)** – RMSE better but MAE worse, or R² < 0.5.
4. **Generated, beat baseline (Moderate)** – RMSE and MAE both better and R² ≥ 0.5 (the 0.5 cut-off is an arbitrary documented constant, `R2_USEFUL`).

Also changed: the “±RMSE” label (which read as an error bound) is now “Model held-out RMSE” with a tooltip stating it is not a prediction interval; capacity/PUE projections from a Low-reliability forecast are no longer generated (they used to appear at capped priority).

## 5. Rule audit

| Rule | Trigger | Priority | Evidence shown | Notes |
|---|---|---|---|---|
| PWR-1 | facility/capacity ≥ 80 / 90 / 95 % | Medium / High / Critical | kW vs capacity, headroom, thresholds | thresholds inclusive (≥), float-safe |
| THM-1 | headroom = limit − inlet: ≤ 0 / < 2 / < 4 °C | Critical / High / Medium | inlet, limit, headroom, rule thresholds | **wording fixed** (below) |
| THM-2 | inlet < 18 °C | Low | inlet vs min | |
| THM-3 / 4 | ΔT < 8 / > 20 °C | Medium | outlet − inlet | |
| THM-5 | cooling status ≠ Normal | High / Critical | status, IT load, headroom | |
| THM-6 | RH outside 20–70 % (Medium) or 8–80 % (High) | Medium / High | RH, range | |
| ENR-1 | PUE ≥ 1.5 / 1.8 | Medium / High | PUE, overhead, illustrative gap to 1.4 | |
| ENR-2 / 3 / 4 | low CPU+mem(+GPU); low GPU with active CPU; any ≥ 90 % | Medium / Low / Medium | values now shown with thresholds | |
| SUS-1 / 2 | carbon ≥ 400 g/kWh and renewable < 50 %; renewable < 30 % | Medium / Low | values with thresholds | 50 % was hard-coded; now configurable (`renewable_limited`) |
| FCT-1 | projected capacity / PUE crosses threshold | up to Critical | relative change, projection, assumptions | **no longer fires for Low reliability** |
| FCT-2 | |forecast − last| > 1 × training IQR | Medium (Info if Low) | last, forecast, ratio | |
| FCT-3 | reliability Low | Info | RMSE and MAE model vs baseline | |
| FCT-4 / 5 | implausible forecast / outside-range or top-bin input | High / Medium | message | |

**Observed case (inlet 24 °C, limit 27 °C):** headroom 3 °C < 4 °C → Medium, so the *trigger* was consistent with the code, but the title “at or near the thermal limit” overstated a 3 °C margin. The title now reads “Inlet temperature is less than 4 °C below the thermal limit” (using the configured margin), the evidence quotes the rule thresholds, and the margins remain editable defaults (sidebar). Operating limits were not changed. Headroom comparisons are strict `<` (exactly 4 °C or exactly 2 °C does not escalate); this is tested and shown in the evidence text.

New: inconsistent threshold sets (e.g. watch > high) are rejected rather than silently disabling rules.

## 6. Energy, cost, emissions

Verified and tested: IT kW = facility kW ÷ entered PUE (valid only if the entered PUE is consistent with the entered facility power; labelled calculated); daily energy = facility kW × 24 h (constant-load estimate); cost = kWh × tariff per kWh; emissions = kWh × gCO₂/kWh ÷ 1000 = kg (the KPI shows tonnes = kg ÷ 1000); renewable share does not affect the location-based emissions figure; capacity percentage = facility power ÷ entered capacity. All are labelled estimates, never measurements. New assumption recorded: the entered capacity must be a facility-level figure. A model-derived % change is applied to facility power only for non-Low forecasts and is labelled illustrative (IT-power signal assumes constant PUE; the modelled series comes from a different site).

## 7. Files changed

| File | Change |
|---|---|
| `nexus_core/forecast.py` | `assess_reliability`; top-bin saturation guard; absurd-value guard; stricter CSV/paste validation; extra result fields |
| `nexus_core/decision.py` | THM-1 wording/evidence; `validate_limits`; float-safe capacity thresholds; configurable renewable threshold; FCT gating; wider input validation |
| `nexus_core/analysis.py` | validation of limits; timestamp assumption; richer JSON export |
| `streamlit_app.py` | metric labels, reliability banner and baseline captions only; layout unchanged |
| `tests/test_app.py`, `tests/test_core.py` | made time-independent; updated for corrected behaviour |
| `tests/test_audit.py` (new) | 53 new tests (plus 1 new and 3 changed tests elsewhere; suite 50 → 104) |
| `tools/audit_model_probe.py`, `AUDIT_PROBE_OUTPUT.txt` (new) | reproducible audit |
| `README.md`, `CHANGELOG.md`, `LIMITATIONS.md`, `TESTING_REPORT.md` | updated |
| Not touched | `models/*`, `requirements.txt` |

## 8. Unresolved issues

1. Recorded metrics not reproduced; no data. Run `tools/verify_against_training_csv.py` on your CSVs before the viva.
2. Lag interval and calendar-feature semantics not verified against training code.
3. The saved model saturates above 1,947 kW for IT power (now warned, not fixed); similar flat behaviour above training range on other signals (Probe D).
4. Stored training ranges describe the first 60 % of each series, not the 80 % the model was fitted on.
5. Model output depends strongly on calendar features: for a constant 2.49 kW CRAC-3 input the forecast ranges from −9.9 to 40.1 kW across UTC hours. This is a synthetic-input observation, not an accuracy result.
6. No seasonal-naive comparison, no prediction intervals, no multi-step evaluation, no candidate model.
7. Pasted readings cannot be checked for order or spacing.
8. Thai meter unit assumed kW.
9. The earlier `TESTING_REPORT.md` claim “50 passed” was not reproduced: the original code gave 49 passed, 1 failed here (a test that used the current clock).

## 9. Run

```bash
cd nexus
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest
python -m pytest -q tests
python tools/audit_model_probe.py
streamlit run streamlit_app.py
```

[![DOI](https://zenodo.org/badge/1373965991.svg)](https://doi.org/10.5281/zenodo.22807397)

# Stage 11.2.1 — End-to-End Analytical Reproducibility Package

This package supports the manuscript **From Managerial Evaluation to AI-Driven Prediction of Employee Performance**. It aligns the processed datasets, temporal split labels, feature-engineering code, and final analytical pipeline with the frozen publication specification.

## Reproducibility scope

### 1. Reproducibility of the final analytical pipeline

Using the three processed longitudinal datasets in `data/`, `src/reproduce_final_analysis.py` reproduces the reported final model metrics, matched test predictions, train-only discrepancy scaling, H1 employee-cluster bootstrap, H2/H3 associations and cluster-bootstrap intervals, discrepancy-index sensitivity analysis, and XGBoost SHAP summaries. Authoritative publication outputs are retained as `results/final_*`; regenerated outputs are written as `results/reproduced_*` and do not overwrite them. Runtime-generated `reproduced_*` files are intentionally excluded from the SHA-256 manifest and are verified numerically against the corresponding frozen `final_*` reference outputs.

### 2. Upstream feature-engineering reproducibility

`src/build_processed_datasets.py` and `src/feature_engineering.py` document and implement the transformation from the **published Factory Workers’ Daily Performance & Attrition raw event-level CSV** to the paired Benchmark/Manager longitudinal feature datasets and target/audit dataset. The transformation uses 30-day observation windows `[t-29,t]`, next-30-day target windows `[t+1,t+30]`, 30-day anchor cadence, and common eligibility thresholds of at least 15 Benchmark efficacy observations, 15 manager-recorded efficacy observations, and 15 target efficacy observations.

The raw source CSV is not redistributed in this package. To rebuild the processed datasets, obtain the published source dataset and run:

`python src/build_processed_datasets.py /path/to/raw_factory_workers.csv`

To verify the complete upstream transformation against the frozen processed inputs used for the publication analysis, run:

`python src/verify_upstream_rebuild.py /path/to/raw_factory_workers.csv`

The verification rebuilds all 8,400 paired employee-windows and compares every processed field against the frozen datasets. The released Stage 11.2.1 package passes this check with no field-level differences. The implementation deliberately preserves the original per-employee pandas/NumPy numerical operations used for efficacy summaries and trends so that tree-model reproduction is not perturbed by otherwise negligible floating-point differences.

### 3. What this package does not reproduce

This package **does not reproduce the original WorkforceSim synthetic data-generation process** that created the published raw event-level dataset. It reproduces the study’s feature-engineering and analytical pipeline *from that published synthetic dataset onward*. Accordingly, “end-to-end” in this repository refers to the study-specific analytical chain, not to regeneration of the underlying synthetic workforce simulation itself.

## Frozen temporal design

The 17 anchors are assigned chronologically as follows:

- anchors 1–9: `train`
- anchor 10: `embargo`
- anchors 11–13: `intermediate_holdout`
- anchor 14: `embargo`
- anchors 15–17: `test`

The `split` column in all three processed datasets records this exact Stage 11.2.1 allocation. There is no legacy `validation` label.

The intermediate holdout is **intentionally unused** in Stage 11.2.1. It is not used for hyperparameter tuning, model selection, early stopping, preprocessing estimation, or final model fitting. It is retained as a temporally separated buffer in the frozen experimental design. The embargo anchors are excluded from model fitting and final evaluation. Final testing uses only anchors 15–17.

This allocation yields 4,448 training employee-windows, 995 embargoed windows, 1,469 intermediate-held-out windows, and 1,488 final-test windows from 511 employees.

## Predictors and target

Six predictors are used in both information conditions:

- `performance_mean_30d`
- `performance_std_30d`
- `performance_trend_30d`
- `attendance_rate_30d`
- `positive_behavior_count_30d`
- `negative_behavior_count_30d`

The target is `future_actual_efficacy_mean_30d`, computed exclusively from simulated actual efficacy observations in `[t+1,t+30]`. Predictors are computed exclusively from `[t-29,t]`. The build code contains temporal-integrity checks enforcing separation between observation and target windows.

`audit_sub_age`, `audit_sub_sex`, and `audit_sup_ID` are audit metadata only and are not predictors. The supervisor identifier is assigned deterministically from the observation window and has no role in the Stage 11.2.1 models or hypotheses.

## Models and preprocessing

The frozen models are Linear Regression, Random Forest, and XGBoost. Model configurations are defined in `config/final_experiment_spec.json`. Train-only median imputation is defined separately for each information condition; the six final predictors contain no missing values in the reported experiment, so no values are effectively imputed.

## Discrepancy and robustness analyses

Absolute Benchmark–Manager discrepancies are standardized using means and population standard deviations estimated only from training anchors 1–9. Attendance discrepancy has zero training variance and therefore contributes zero after standardization. The composite discrepancy index is post-estimation only and is never supplied to the predictive models.

Stage 11.2.1 includes 5,000-replicate employee-cluster bootstrap confidence intervals for H1 and for H2/H3 Pearson and Spearman associations. A sensitivity check excluding the zero-variance attendance dimension yields only a positive linear rescaling of the discrepancy index and leaves the association coefficients unchanged.

## Interpretation of predictive performance

The synthetic dataset exhibits substantial temporal persistence in efficacy, and recent mean performance is the dominant XGBoost feature attribution. The absolute predictive performance reported by the study may therefore partly reflect temporal dependencies embedded in the synthetic data-generating process. This is not target leakage: the target window begins after the observation window. The reported predictive accuracy should not be extrapolated to real employees or organizations.

## Main files

- `src/feature_engineering.py` — longitudinal Benchmark/Manager feature and target construction
- `src/build_processed_datasets.py` — command-line upstream build entry point
- `src/verify_upstream_rebuild.py` — exact upstream rebuild verification against the frozen processed inputs
- `src/reproduce_final_analysis.py` — final Stage 11.2.1 analysis reproduction
- `config/final_experiment_spec.json` — frozen scientific configuration
- `data/` — processed paired datasets with Stage 11.2.1 split labels
- `results/final_*` — frozen authoritative publication outputs, including the temporal-persistence audit used to contextualize absolute predictive performance
- `results/reproduced_*` — runtime-generated analytical outputs; intentionally absent from the distributed package and SHA-256 manifest until reproduction is run
- `src/verify_reproduced_outputs.py` — numerical/structural comparison of regenerated outputs against frozen `final_*` references
- `src/verify_manifest.py` — SHA-256 integrity check for immutable distributed artifacts
- `requirements.lock.txt` — locked Python dependencies
- `MANIFEST_SHA256.csv` — integrity manifest for immutable distributed artifacts only

## Integrity manifest and generated outputs

`MANIFEST_SHA256.csv` authenticates the immutable artifacts distributed with the release: source code, configuration, processed analytical inputs, documentation, and frozen `results/final_*` reference outputs. Files generated at runtime (`results/reproduced_*` and any reproduction warning file) are deliberately excluded because a legitimate reproduction run rewrites or creates them.

This separates two checks that serve different purposes. Run `python src/verify_manifest.py` to verify release integrity. Run `python src/reproduce_final_analysis.py` to regenerate the analytical outputs, followed by `python src/verify_reproduced_outputs.py` to compare those outputs with the frozen publication references. A valid reproduction run must not invalidate the manifest.

## Interpretation and use

The study is a controlled proof of concept using a synthetic dataset. Simulated actual efficacy is an experimental reference exposed by the source simulation, not an objective real-world measure of employee performance. The package and models are not intended for autonomous employment decisions.

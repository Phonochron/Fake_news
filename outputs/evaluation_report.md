# Evaluation report

Selected model: **Linear SVM** (validation F1: 0.9920; held-out test F1: 0.9916).

## Method

- Random seed: 42. Stratified split after text deduplication: train 27259, validation 5841, test 5842.
- TF-IDF is fitted inside each of the 3 classical-model cross-validation folds. Hyperparameters use training folds only.
- The Naive Bayes CV score is the best score from its alpha search. The ensemble uses that selected alpha during CV. These CV scores are diagnostic, not unbiased nested-CV estimates.
- LSTM early stopping uses a subset of the training split. The separate validation split selects one model by F1.
- The test split is evaluated only after selection. All five models are reported for comparison; the test results do not change the selected model.
- Duplicate normalized articles and conflicting-label articles are removed before splitting; the explicit Reuters token is removed during preprocessing.

## Dataset audit

- Raw rows: 44898; unique usable rows: 38942; duplicate normalized rows: 5947; conflicting-label rows: 0.
- Reuters marker rate before preprocessing: real 99.8%, fake 1.3%.
- Subject categories are exclusive to one label: True. Subject and date are excluded from model features.
- Date range: 2015-03-31 to 2018-02-19; unparsed dates: 10.

## Interpretation

The dataset has strong source/style cues even after removing the explicit Reuters token. These scores measure performance on this dataset's random holdout, not independent fact checking or generalization to unseen publishers. No reliable publisher identifier is available for a source-held-out evaluation.
The top-term table ranks Naive Bayes log likelihood differences between HOAX and REAL classes. It is a dataset-level association, not an explanation of an individual prediction or the selected Linear SVM model.

See `data_audit.json`, `subject_label_counts.csv`, `cv_metrics.csv`, `validation_metrics.csv`, and `metrics.csv` for the underlying counts and scores.

## Final decision rule

The selected model scored 0.9920 F1 on validation. The legacy five-model majority scored 0.9871 F1 on the same validation split. The app uses the selected model. Voting Ensemble already combines three of the other votes, so a final majority counts those correlated models again.

See `decision_policy.json` and `training_manifest.json` for the saved rule and provenance of this run.

"""NAMRC 2027: grouped validation of NASA Milling flank-wear prediction.

Input: public/api/experiments.json (created by process_nasa.py)
Outputs: public/api/model_results.json and public/api/model_predictions.json
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / 'public' / 'api'
SOURCE = API / 'experiments.json'
FEATURES = ['time', 'DOC', 'feed', 'material'] + [
    f'{channel}_{stat}'
    for channel in ('smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle')
    for stat in ('mean', 'std', 'rms')
]
RANDOM_STATE = 42


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def metrics(y_true, y_pred):
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None,
    }


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Wrote', path.relative_to(ROOT))


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Missing {SOURCE}. Run scripts/process_nasa.py first.')
    payload = json.loads(SOURCE.read_text(encoding='utf-8'))
    records = payload['records']
    if not isinstance(records, list):
        raise ValueError('records must be a list')
    labeled = [r for r in records if finite_number(r.get('VB')) is not None]
    unlabeled = len(records) - len(labeled)
    if not labeled:
        raise ValueError('No measured VB labels found')
    # Flag grossly anomalous sensor summaries; do not silently fit on them.
    # This is a provisional numeric-integrity threshold, not a physical sensor specification.
    flagged = [r for r in labeled if any(
        abs(float(r.get(f, 0))) > 1e6 for f in FEATURES[4:]
        if finite_number(r.get(f)) is not None
    )]
    flagged_ids = [int(r['experiment_index']) for r in flagged]
    labeled = [r for r in labeled if int(r['experiment_index']) not in flagged_ids]
    print('Flagged extreme sensor-summary records:', flagged_ids)
    groups = np.array([int(r['case']) for r in labeled])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        raise ValueError('At least five distinct cases required for 5-fold GroupKFold')
    X = np.array([[finite_number(r.get(f)) if finite_number(r.get(f)) is not None else np.nan
                   for f in FEATURES] for r in labeled], dtype=float)
    y = np.array([float(r['VB']) for r in labeled], dtype=float)
    if np.any(np.all(np.isnan(X), axis=0)):
        missing = [FEATURES[i] for i in np.where(np.all(np.isnan(X), axis=0))[0]]
        raise ValueError(f'Entirely missing features: {missing}')

    models = {
        'Mean baseline': DummyRegressor(strategy='mean'),
        'Multiple Linear Regression': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LinearRegression()),
        'Ridge Regression': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=10.0)),
        'Random Forest': make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=1)),
    }
    cv = GroupKFold(n_splits=5)
    folds = list(cv.split(X, y, groups))
    results = {}
    predictions = {}
    fold_description = []
    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        fold_description.append({
            'fold': fold_id,
            'train_cases': sorted(set(int(v) for v in groups[train_idx])),
            'validation_cases': sorted(set(int(v) for v in groups[test_idx])),
            'training_records': len(train_idx),
            'validation_records': len(test_idx),
        })
    for name, estimator in models.items():
        oof = np.full(len(y), np.nan, dtype=float)
        per_fold = []
        for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
            model = clone(estimator)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            oof[test_idx] = pred
            per_fold.append({'fold': fold_id, **metrics(y[test_idx], pred)})
        if not np.isfinite(oof).all():
            raise RuntimeError(f'Incomplete out-of-fold predictions for {name}')
        results[name] = {'out_of_fold_metrics': metrics(y, oof), 'fold_metrics': per_fold}
        predictions[name] = oof
        print(name, results[name]['out_of_fold_metrics'])

    # Select the lowest OOF MAE; this is exploratory model selection, not an independent test estimate.
    best = min(results, key=lambda name: results[name]['out_of_fold_metrics']['MAE'])
    output_rows = []
    for i, row in enumerate(labeled):
        output_rows.append({
            'experiment_index': int(row['experiment_index']),
            'case': int(row['case']),
            'run': int(row['run']),
            'observed_VB': float(y[i]),
            'out_of_fold_predictions': {name: float(pred[i]) for name, pred in predictions.items()},
        })
    meta = {
        'dataset': 'NASA Milling',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': payload.get('metadata', {}).get('sha256'),
        'target': 'VB',
        'target_units': 'verify against NASA Readme.pdf',
        'record_count': len(records),
        'labeled_records_before_quality_filter': len(labeled) + len(flagged),
        'labeled_records_used': len(labeled),
        'excluded_extreme_sensor_records': flagged_ids,
        'quality_filter': 'Provisional: exclude records with any sensor mean/std/RMS absolute value > 1e6; verify raw signal and NASA documentation',
        'unlabeled_records': unlabeled,
        'distinct_cases': len(unique_groups),
        'feature_names': FEATURES,
        'excluded_from_features': ['VB', 'case', 'run', 'experiment_index', 'source_file', 'source_variable', '*_samples'],
        'validation': '5-fold GroupKFold by case; out-of-fold predictions',
        'random_state': RANDOM_STATE,
        'limitations': [
            'Model selection uses the same grouped cross-validation results; the selected-model score is not an independent test-set estimate.',
            'No model hyperparameter tuning was performed.',
            'Extreme sensor-summary records were provisionally excluded pending raw-signal investigation.',
            'Time is included as a predictor; predictions require that elapsed machining time is known at inference.',
            'Sensor summary statistics are computed from each full run, so this is run-level estimation, not early warning from partial runs.',
            'Prediction outputs may be negative because regressors are not constrained to nonnegative wear.',
        ],
    }
    save(API / 'model_results.json', {
        'metadata': meta,
        'folds': fold_description,
        'models': results,
        'exploratory_lowest_MAE_model': best,
    })
    save(API / 'model_predictions.json', {'metadata': meta, 'records': output_rows})
    print('Finished. Exploratory lowest MAE:', best)


if __name__ == '__main__':
    main()

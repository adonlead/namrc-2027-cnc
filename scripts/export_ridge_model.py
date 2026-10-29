"""NAMRC 2027: export full-data Ridge model for browser inference.

Run after process_nasa.py and train_model.py. This is a deployable fitted
model, NOT an independent validation result. Never train on unlabeled runs.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_model import API, FEATURES, ROOT, finite_number, save


def main():
    source = API / 'experiments.json'
    results_path = API / 'model_results.json'
    if not source.exists() or not results_path.exists():
        raise FileNotFoundError('Run process_nasa.py and train_model.py first.')
    data = json.loads(source.read_text(encoding='utf-8'))
    results = json.loads(results_path.read_text(encoding='utf-8'))
    if results['metadata']['feature_names'] != FEATURES:
        raise ValueError('Feature ordering differs from validation model.')
    records = data['records']
    labeled = [r for r in records if finite_number(r.get('VB')) is not None]
    excluded = set(int(v) for v in results['metadata']['excluded_extreme_sensor_records'])
    filtered = [r for r in labeled if int(r['experiment_index']) not in excluded]
    if len(filtered) != results['metadata']['labeled_records_used']:
        raise ValueError('Training record count differs from validated record count.')
    X = np.asarray([[finite_number(r.get(f)) if finite_number(r.get(f)) is not None else np.nan
                     for f in FEATURES] for r in filtered], dtype=float)
    y = np.asarray([float(r['VB']) for r in filtered], dtype=float)
    if np.any(np.all(np.isnan(X), axis=0)):
        raise ValueError('At least one feature is entirely missing.')
    pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=10.0))
    pipeline.fit(X, y)
    imputer, scaler, ridge = pipeline.steps[0][1], pipeline.steps[1][1], pipeline.steps[2][1]
    if len(imputer.statistics_) != len(FEATURES):
        raise ValueError('Imputer removed a feature unexpectedly.')
    # The JS prediction formula is: intercept + sum(coef[j] * ((x[j] or median[j]) - mean[j]) / scale[j]).
    payload = {
        'metadata': {
            'dataset': 'NASA Milling',
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'source_sha256': results['metadata']['source_sha256'],
            'model': 'Ridge Regression',
            'alpha': 10.0,
            'target': 'VB',
            'target_units': 'verify against NASA Readme.pdf',
            'training_records': len(filtered),
            'excluded_experiment_indices': sorted(excluded),
            'training_policy': 'Fit on all quality-screened labeled records after exploratory model comparison',
            'validation_note': 'See model_results.json for grouped out-of-fold exploratory CV; this artifact is fitted on all eligible labeled data and is not an independent test.',
            'inference_scope': 'Run-level estimate requiring elapsed machining time and full-run sensor statistics; no arbitrary cutting-condition-only forecasts',
            'feature_count': len(FEATURES),
        },
        'features': FEATURES,
        'preprocessing': {
            'missing_value_policy': 'Missing/null feature -> training median; reject nonfinite values and out-of-range input',
            'imputer_medians': [float(v) for v in imputer.statistics_],
            'scaler_means': [float(v) for v in scaler.mean_],
            'scaler_scales': [float(v) for v in scaler.scale_],
            'observed_min': [float(v) for v in np.nanmin(X, axis=0)],
            'observed_max': [float(v) for v in np.nanmax(X, axis=0)],
        },
        'ridge': {
            'intercept': float(ridge.intercept_),
            'coefficients': [float(v) for v in ridge.coef_],
        },
    }
    # Verify that the portable formula exactly reproduces sklearn predictions.
    X_imp = np.where(np.isnan(X), np.asarray(payload['preprocessing']['imputer_medians']), X)
    reconstructed = payload['ridge']['intercept'] + (
        (X_imp - np.asarray(payload['preprocessing']['scaler_means'])) /
        np.asarray(payload['preprocessing']['scaler_scales'])
    ) @ np.asarray(payload['ridge']['coefficients'])
    expected = pipeline.predict(X)
    error = float(np.max(np.abs(reconstructed - expected)))
    if error > 1e-10:
        raise AssertionError(f'Portable prediction mismatch: {error}')
    payload['verification'] = {
        'max_absolute_reconstruction_difference': error,
        'tolerance': 1e-10,
        'status': 'passed',
        'sample_checks': [
            {'experiment_index': int(filtered[i]['experiment_index']),
             'sklearn_prediction': float(expected[i]),
             'portable_prediction': float(reconstructed[i])}
            for i in sorted(set([0, len(filtered)//2, len(filtered)-1]))
        ],
    }
    save(API / 'ridge_model.json', payload)
    print('Exported Ridge model; training records:', len(filtered), '; max reconstruction error:', error)


if __name__ == '__main__':
    main()

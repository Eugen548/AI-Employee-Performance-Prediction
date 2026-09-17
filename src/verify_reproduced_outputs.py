from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'
# All runtime analytical outputs expected from reproduce_final_analysis.py.
names=[
'cluster_bootstrap.csv','cluster_bootstrap_H2_H3.csv','discrepancy_analysis.csv',
'discrepancy_associations.csv','discrepancy_index_sensitivity.csv',
'discrepancy_train_scaling_parameters.csv','model_metrics.csv',
'temporal_persistence_audit.csv','test_predictions.csv','xgb_shap_comparison.csv',
'xgb_shap_global.csv']
# Floating-point serialization and library-level SHAP differences can occur at tiny scale.
ATOL=1e-6; RTOL=1e-6
fail=[]
for name in names:
    fp=(R/'discrepancy_train_scaling_parameters.csv') if name=='discrepancy_train_scaling_parameters.csv' else R/f'final_{name}'
    rp=R/f'reproduced_{name}'
    if not fp.exists() or not rp.exists():
        fail.append((name,'MISSING')); continue
    a=pd.read_csv(fp); b=pd.read_csv(rp)
    if list(a.columns)!=list(b.columns) or a.shape!=b.shape:
        fail.append((name,'STRUCTURE')); continue
    ok=True
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if not np.allclose(a[c].to_numpy(float),b[c].to_numpy(float),rtol=RTOL,atol=ATOL,equal_nan=True): ok=False; break
        else:
            if not a[c].fillna('<NA>').astype(str).equals(b[c].fillna('<NA>').astype(str)): ok=False; break
    print(f'{name}: {"PASS" if ok else "FAIL"}')
    if not ok: fail.append((name,'VALUES'))
if fail:
    print(f'REPRODUCTION VERIFICATION: FAIL ({len(fail)} output(s))')
    sys.exit(1)
print('REPRODUCTION VERIFICATION: PASS')

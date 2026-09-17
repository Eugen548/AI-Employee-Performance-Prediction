from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from feature_engineering import BuildConfig, load_raw_csv, build_longitudinal_dataset

ROOT=Path(__file__).resolve().parents[1]
FILES=['benchmark_features_v1.csv','manager_features_v1.csv','targets_audit_v1.csv']

def equal_frames(a,b):
    if list(a.columns)!=list(b.columns) or a.shape!=b.shape:
        return False, {'shape_or_columns': True}
    diffs={}
    for c in a.columns:
        x,y=a[c],b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            mask=~(x.isna() & y.isna())
            ok=np.isclose(x[mask].to_numpy(float),y[mask].to_numpy(float),rtol=1e-12,atol=1e-12,equal_nan=True)
            if not ok.all(): diffs[c]=int((~ok).sum())
        else:
            ok=x.fillna('<NA>').astype(str).eq(y.fillna('<NA>').astype(str))
            if not ok.all(): diffs[c]=int((~ok).sum())
    return not diffs, diffs

def main():
    ap=argparse.ArgumentParser(description='Rebuild processed datasets from the raw published CSV and verify equality with the frozen Stage 11.2.1 data.')
    ap.add_argument('raw_csv',type=Path)
    args=ap.parse_args()
    raw=load_raw_csv(args.raw_csv)
    rebuilt=build_longitudinal_dataset(raw,BuildConfig(use_extended_features=False))
    ok_all=True
    for fn,df in zip(FILES,rebuilt):
        frozen=pd.read_csv(ROOT/'data'/fn)
        # normalize dates through CSV serialization, exactly as the public build command does
        tmp=df.copy()
        for c in ['anchor_date','obs_start','obs_end','target_start','target_end']:
            if c in tmp: tmp[c]=tmp[c].astype(str)
        ok,diffs=equal_frames(frozen,tmp)
        print(f'{fn}: {"PASS" if ok else "FAIL"}; rows={len(tmp):,}; differences={diffs}')
        ok_all &= ok
    if not ok_all:
        raise SystemExit(1)
    print('UPSTREAM REBUILD VERIFICATION: PASS')

if __name__=='__main__': main()

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from feature_engineering import BuildConfig, load_raw_csv, build_longitudinal_dataset

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description='Build Stage 11.2.1 processed longitudinal datasets from the published raw synthetic event CSV.')
    ap.add_argument('raw_csv', type=Path, help='Path to Factory Workers raw event-level CSV')
    ap.add_argument('--output-dir', type=Path, default=ROOT/'rebuilt_data', help='Destination directory')
    args=ap.parse_args()
    df=load_raw_csv(args.raw_csv)
    B,M,T=build_longitudinal_dataset(df, BuildConfig(use_extended_features=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    B.to_csv(args.output_dir/'benchmark_features_v1.csv', index=False)
    M.to_csv(args.output_dir/'manager_features_v1.csv', index=False)
    T.to_csv(args.output_dir/'targets_audit_v1.csv', index=False)
    print(f'Raw rows: {len(df):,}; employees: {df.sub_ID.nunique():,}')
    print(f'Generated paired windows: {len(T):,}; anchors: {T.anchor_date.nunique()}')
    print(T['split'].value_counts().to_string())

if __name__=='__main__': main()

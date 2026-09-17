from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

POSITIVE = {"Idea", "Feat", "Teamwork", "Sacrifice"}
NEGATIVE = {"Lapse", "Slip", "Disruption", "Sabotage"}
BEHAVIORS = sorted(POSITIVE | NEGATIVE)
CORE_FEATURES = [
    "performance_mean_30d", "performance_std_30d", "performance_trend_30d",
    "attendance_rate_30d", "positive_behavior_count_30d", "negative_behavior_count_30d",
]
REQUIRED_COLUMNS = {
    "sub_ID", "event_date", "behav_comptype_h", "actual_efficacy_h",
    "record_comptype", "recorded_efficacy"
}

@dataclass
class BuildConfig:
    observation_days: int = 30
    target_days: int = 30
    anchor_step_days: int = 30
    min_obs_benchmark: int = 15
    min_obs_manager: int = 15
    min_target_obs: int = 15
    use_extended_features: bool = False

def load_raw_csv(source) -> pd.DataFrame:
    try:
        df = pd.read_csv(source, encoding="cp1252", low_memory=False)
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, encoding="utf-8", low_memory=False)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["_source_order"] = np.arange(len(df), dtype=np.int64)
    return df

def validate_columns(df: pd.DataFrame) -> list[str]:
    return sorted(REQUIRED_COLUMNS.difference(df.columns))

def dataset_audit(df: pd.DataFrame) -> dict:
    missing = validate_columns(df)
    return {
        "rows": len(df), "columns": len(df.columns),
        "employees": int(df["sub_ID"].nunique()) if "sub_ID" in df else None,
        "date_min": df["event_date"].min() if "event_date" in df else None,
        "date_max": df["event_date"].max() if "event_date" in df else None,
        "duplicates": int(df.drop(columns=["_source_order"], errors="ignore").duplicated().sum()),
        "missing_required_columns": missing,
    }

def _efficacy_summary(eff: pd.DataFrame, value_col: str, obs_start: pd.Timestamp) -> pd.DataFrame:
    # Deliberately use the same per-employee pandas/NumPy operations as the original
    # research pipeline. This preserves floating-point values exactly enough for
    # deterministic Random Forest split reproduction.
    rows=[]
    for sid,g in eff.groupby("sub_ID", sort=False):
        vals=pd.to_numeric(g[value_col], errors="coerce")
        valid=vals.notna()
        gv=g.loc[valid]
        y=vals.loc[valid].to_numpy(float)
        x=(gv["event_date"]-obs_start).dt.days.to_numpy(float)
        trend=np.nan
        if len(y)>=2 and not np.all(x==x[0]):
            trend=float(np.polyfit(x,y,1)[0]*30.0)
        rows.append((sid,int(valid.sum()),float(vals.mean()),float(vals.std(ddof=1)),trend))
    return pd.DataFrame(rows,columns=["sub_ID","n","mean","std","trend"]).set_index("sub_ID")

def _event_summary(obs: pd.DataFrame, event_col: str) -> pd.DataFrame:
    ct = obs.groupby(["sub_ID", event_col], sort=False).size().unstack(fill_value=0)
    idx = ct.index
    def col(name):
        return ct[name] if name in ct.columns else pd.Series(0, index=idx, dtype=float)
    presence, absence = col("Presence"), col("Absence")
    denom = presence + absence
    out = pd.DataFrame(index=idx)
    out["attendance"] = (presence / denom.replace(0, np.nan)).astype(float)
    for k in BEHAVIORS:
        out[k] = col(k).astype(int)
    out["positive"] = sum(out[k] for k in POSITIVE)
    out["negative"] = sum(out[k] for k in NEGATIVE)
    return out

def _view_table(obs: pd.DataFrame, eff: pd.DataFrame, view: str, obs_start: pd.Timestamp) -> pd.DataFrame:
    value_col = "actual_efficacy_h" if view == "benchmark" else "recorded_efficacy"
    event_col = "behav_comptype_h" if view == "benchmark" else "record_comptype"
    es = _efficacy_summary(eff, value_col, obs_start)
    ev = _event_summary(obs, event_col)
    q = es.join(ev, how="left")
    out = pd.DataFrame(index=q.index)
    out["performance_mean_30d"] = q["mean"]
    out["performance_std_30d"] = q["std"]
    out["performance_trend_30d"] = q["trend"]
    out["attendance_rate_30d"] = q["attendance"]
    out["positive_behavior_count_30d"] = q["positive"]
    out["negative_behavior_count_30d"] = q["negative"]
    for src, dst in [("Idea","idea_count_30d"),("Feat","feat_count_30d"),("Teamwork","teamwork_count_30d"),
                     ("Sacrifice","sacrifice_count_30d"),("Lapse","lapse_count_30d"),("Slip","slip_count_30d"),
                     ("Disruption","disruption_count_30d"),("Sabotage","sabotage_count_30d")]:
        out[dst] = q[src]
    out["n_efficacy_obs"] = q["n"].astype(int)
    return out

def _audit_metadata(obs: pd.DataFrame, eligible: list) -> pd.DataFrame:
    o = obs[obs["sub_ID"].isin(eligible)]
    latest = o.sort_values(["event_date","_source_order"]).groupby("sub_ID", sort=False).tail(1).set_index("sub_ID")
    result = pd.DataFrame(index=pd.Index(eligible, name="sub_ID"))
    result["audit_sub_age"] = latest.reindex(eligible)["sub_age"] if "sub_age" in latest else np.nan
    result["audit_sub_sex"] = latest.reindex(eligible)["sub_sex"] if "sub_sex" in latest else np.nan
    if "sup_ID" not in o.columns:
        result["audit_sup_ID"] = np.nan
        return result
    sv = o.dropna(subset=["sup_ID"])
    if sv.empty:
        result["audit_sup_ID"] = np.nan
        return result
    counts = sv.groupby(["sub_ID","sup_ID"], sort=False).size().rename("n").reset_index()
    mx = counts.groupby("sub_ID")["n"].transform("max")
    modes = counts[counts["n"].eq(mx)][["sub_ID","sup_ID"]]
    cand = sv.merge(modes, on=["sub_ID","sup_ID"], how="inner")
    chosen = cand.sort_values(["event_date","_source_order"]).groupby("sub_ID", sort=False).tail(1).set_index("sub_ID")["sup_ID"]
    result["audit_sup_ID"] = chosen.reindex(eligible)
    return result

def build_longitudinal_dataset(df: pd.DataFrame, cfg: BuildConfig):
    missing = validate_columns(df)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "_source_order" not in df.columns:
        df = df.copy(); df["_source_order"] = np.arange(len(df), dtype=np.int64)

    min_date, max_date = df["event_date"].min(), df["event_date"].max()
    first_anchor = min_date + pd.Timedelta(days=cfg.observation_days - 1)
    last_anchor = max_date - pd.Timedelta(days=cfg.target_days)
    anchors = list(pd.date_range(first_anchor, last_anchor, freq=f"{cfg.anchor_step_days}D"))
    B, M, T = [], [], []

    for anchor in anchors:
        obs_start = anchor - pd.Timedelta(days=cfg.observation_days - 1)
        tar_start, tar_end = anchor + pd.Timedelta(days=1), anchor + pd.Timedelta(days=cfg.target_days)
        obs = df[df["event_date"].between(obs_start, anchor)]
        tar = df[df["event_date"].between(tar_start, tar_end)]
        obs_eff = obs[obs["behav_comptype_h"].eq("Efficacy")]
        tar_eff = tar[tar["behav_comptype_h"].eq("Efficacy")]

        bsum = _view_table(obs, obs_eff, "benchmark", obs_start)
        msum = _view_table(obs, obs_eff, "manager", obs_start)
        target_rows=[]
        for sid,g in tar_eff.groupby("sub_ID", sort=False):
            vals=pd.to_numeric(g["actual_efficacy_h"], errors="coerce")
            target_rows.append((sid,int(vals.notna().sum()),float(vals.mean())))
        tsum=pd.DataFrame(target_rows,columns=["sub_ID","count","mean"]).set_index("sub_ID")
        eligible = sorted(set(bsum.index[bsum["n_efficacy_obs"] >= cfg.min_obs_benchmark]) &
                          set(msum.index[msum["n_efficacy_obs"] >= cfg.min_obs_manager]) &
                          set(tsum.index[tsum["count"] >= cfg.min_target_obs]))
        audit = _audit_metadata(obs, eligible)

        common = pd.DataFrame({
            "window_id": [f"{sid}_{anchor.date()}" for sid in eligible], "sub_ID": eligible,
            "anchor_date": anchor, "obs_start": obs_start, "obs_end": anchor,
            "target_start": tar_start, "target_end": tar_end,
        }).set_index("sub_ID", drop=False)
        B.append(pd.concat([common, bsum.reindex(eligible)], axis=1).loc[:,~pd.concat([common, bsum.reindex(eligible)], axis=1).columns.duplicated()])
        M.append(pd.concat([common, msum.reindex(eligible)], axis=1).loc[:,~pd.concat([common, msum.reindex(eligible)], axis=1).columns.duplicated()])
        tt = common.copy()
        tt["future_actual_efficacy_mean_30d"] = tsum.reindex(eligible)["mean"].to_numpy()
        tt["n_target_efficacy_obs"] = tsum.reindex(eligible)["count"].astype(int).to_numpy()
        for c in ["audit_sub_age","audit_sub_sex","audit_sup_ID"]:
            tt[c] = audit.reindex(eligible)[c].to_numpy()
        T.append(tt)

    B, M, T = (pd.concat(x, ignore_index=True) for x in (B, M, T))
    dates = sorted(pd.to_datetime(T["anchor_date"]).unique())
    if len(dates) != 17:
        raise ValueError(f"Stage 11.2.1 expects 17 anchors, found {len(dates)}")
    amap = {pd.Timestamp(a): i+1 for i,a in enumerate(dates)}
    anum = pd.to_datetime(T["anchor_date"]).map(amap)
    smap = {**{i:"train" for i in range(1,10)}, 10:"embargo", **{i:"intermediate_holdout" for i in range(11,14)},
            14:"embargo", **{i:"test" for i in range(15,18)}}
    split = anum.map(smap).to_numpy()
    for d in (B,M,T): d["split"] = split

    assert (pd.to_datetime(T["obs_end"]) < pd.to_datetime(T["target_start"])).all()
    assert ((pd.to_datetime(T["target_start"]) - pd.to_datetime(T["obs_end"])).dt.days == 1).all()
    return B, M, T

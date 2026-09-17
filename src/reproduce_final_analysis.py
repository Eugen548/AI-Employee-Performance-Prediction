from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
spec=json.loads((ROOT/'config'/'final_experiment_spec.json').read_text())
B=pd.read_csv(ROOT/'data'/'benchmark_features_v1.csv')
M=pd.read_csv(ROOT/'data'/'manager_features_v1.csv')
T=pd.read_csv(ROOT/'data'/'targets_audit_v1.csv')

# Alignment checks
assert B['window_id'].equals(M['window_id']) and B['window_id'].equals(T['window_id'])
anchors=sorted(pd.to_datetime(T.anchor_date).unique())
amap={pd.Timestamp(a):i+1 for i,a in enumerate(anchors)}
anum=pd.to_datetime(T.anchor_date).map(amap)
train=anum.isin(spec['split']['train_anchors'])
holdout=anum.isin(spec['split']['intermediate_holdout_anchors'])
test=anum.isin(spec['split']['test_anchors'])
embargo=anum.isin(spec['split']['embargo_anchors'])
expected_split=np.select([train,embargo,holdout,test],['train','embargo','intermediate_holdout','test'],default='UNASSIGNED')
for D,label in [(B,'Benchmark'),(M,'Manager'),(T,'Targets')]:
    if 'split' not in D.columns or not np.array_equal(D['split'].astype(str).to_numpy(),expected_split):
        raise ValueError(f'{label} split column does not match frozen Stage 11.2.1 anchor allocation')
features=spec['features']; target=spec['target']; y=T[target].astype(float)

def prep(D):
    X=D[features].astype(float).copy()
    med=X.loc[train].median(numeric_only=True)
    return X.fillna(med)
XB=prep(B); XM=prep(M)

# Fresh estimator per view/model.
def make_models():
    return {
      'Linear Regression':LinearRegression(),
      'Random Forest':RandomForestRegressor(n_estimators=500,min_samples_leaf=2,random_state=42,n_jobs=-1),
      'XGBoost':XGBRegressor(n_estimators=500,learning_rate=.03,max_depth=4,subsample=.9,colsample_bytree=.9,objective='reg:squarederror',random_state=42,n_jobs=-1)
    }

pred={}; metric_rows=[]; fitted={}
for view,X in [('Benchmark',XB),('Manager',XM)]:
    for name,model in make_models().items():
        model.fit(X.loc[train],y.loc[train]); fitted[(view,name)]=model
        p=np.asarray(model.predict(X.loc[test]),dtype=float); pred[(view,name)]=p
        metric_rows.append({'View':view,'Model':name,'N_train':int(train.sum()),'N_intermediate_holdout':int(holdout.sum()),'N_test':int(test.sum()),'MAE':mean_absolute_error(y.loc[test],p),
          'RMSE':np.sqrt(mean_squared_error(y.loc[test],p)),'R2':r2_score(y.loc[test],p)})
pd.DataFrame(metric_rows).to_csv(OUT/'reproduced_model_metrics.csv',index=False)

# Temporal-persistence audit used only to contextualize absolute predictive performance.
persistence=[]
for view,D in [('Benchmark',B),('Manager',M)]:
    previous_mean=D.loc[test,'performance_mean_30d'].astype(float).to_numpy()
    target_test=y.loc[test].to_numpy()
    persistence.append({
        'View':view,
        'Pearson_previous30d_mean_vs_future30d_target':pearsonr(previous_mean,target_test).statistic,
        'Persistence_MAE':mean_absolute_error(target_test,previous_mean),
        'Persistence_RMSE':np.sqrt(mean_squared_error(target_test,previous_mean)),
        'Persistence_R2':r2_score(target_test,previous_mean),
        'N_test':len(target_test)
    })
pd.DataFrame(persistence).to_csv(OUT/'reproduced_temporal_persistence_audit.csv',index=False)

# Matched predictions
idx=np.where(test)[0]; yt=y.loc[test].to_numpy()
P=T.loc[test,['window_id','sub_ID','anchor_date','audit_sub_age','audit_sub_sex','audit_sup_ID',target]].copy()
P['anchor_number']=anum.loc[test].to_numpy()
# put anchor_number after anchor_date
cols=['window_id','sub_ID','anchor_date','anchor_number','audit_sub_age','audit_sub_sex','audit_sup_ID',target]
P=P[cols]
for view in ['Benchmark','Manager']:
    vl=view.lower()
    for name in ['Linear Regression','Random Forest','XGBoost']:
        key=name.replace(' ','_')
        p=pred[(view,name)]
        P[f'{key}_{vl}_pred']=p
        P[f'{key}_{vl}_abs_error']=np.abs(yt-p)
P.to_csv(OUT/'reproduced_test_predictions.csv',index=False)

# Train-only discrepancy scaling
Dtrain=(XM.loc[train,features]-XB.loc[train,features]).abs()
mu=Dtrain.mean(axis=0); sd=Dtrain.std(axis=0,ddof=0)
scale=pd.DataFrame({'Feature':features,'Train_absdiff_mean':[mu[f] for f in features],
                    'Train_absdiff_population_sd':[sd[f] for f in features],
                    'Zero_variance':[bool(sd[f]==0) for f in features]})
scale.to_csv(OUT/'reproduced_discrepancy_train_scaling_parameters.csv',index=False)

Dt=(XM.loc[test,features]-XB.loc[test,features])
Da=Dt.abs(); Z=pd.DataFrame(index=Da.index)
for f in features:
    Z[f]=0.0 if sd[f]==0 else (Da[f]-mu[f])/sd[f]
DI=Z.mean(axis=1).to_numpy()

A=T.loc[test,['window_id','sub_ID','anchor_date','audit_sub_age','audit_sub_sex','audit_sup_ID',target]].copy()
for f in features:
    A[f'diff_{f}']=Dt[f].to_numpy(); A[f'absdiff_{f}']=Da[f].to_numpy()
A['discrepancy_index_z']=DI
A['behavior_information_loss']=(Da['positive_behavior_count_30d']+Da['negative_behavior_count_30d']).to_numpy()
assoc=[]
for name in ['Linear Regression','Random Forest','XGBoost']:
    key=name.replace(' ','_')
    div=np.abs(pred[('Manager',name)]-pred[('Benchmark',name)])
    deg=np.abs(yt-pred[('Manager',name)])-np.abs(yt-pred[('Benchmark',name)])
    A[f'{key}_abs_prediction_divergence']=div; A[f'{key}_error_degradation']=deg
    for outcome,vals in [(f'{key}_abs_prediction_divergence',div),(f'{key}_error_degradation',deg)]:
        pr,pp=pearsonr(DI,vals); sr,sp=spearmanr(DI,vals)
        assoc.append({'Model':name,'Outcome':outcome,'Pearson_r':pr,'Pearson_p':pp,
                      'Spearman_rho':sr,'Spearman_p':sp,'N':len(vals)})
A.to_csv(OUT/'reproduced_discrepancy_analysis.csv',index=False)
pd.DataFrame(assoc).to_csv(OUT/'reproduced_discrepancy_associations.csv',index=False)

# Employee-cluster bootstrap of mean paired absolute-error degradation
rng=np.random.default_rng(20260914)
emp=P['sub_ID'].to_numpy(); employees=np.unique(emp)
boot=[]
for name in ['Linear Regression','Random Forest','XGBoost']:
    deg=np.abs(yt-pred[('Manager',name)])-np.abs(yt-pred[('Benchmark',name)])
    groups={e:deg[emp==e] for e in employees}
    vals=np.empty(5000)
    for b in range(5000):
        sampled=rng.choice(employees,size=len(employees),replace=True)
        vals[b]=np.concatenate([groups[e] for e in sampled]).mean()
    boot.append({'Model':name,'Mean_error_degradation':deg.mean(),
                 'CI95_low':np.quantile(vals,.025),'CI95_high':np.quantile(vals,.975),
                 'N_windows':len(deg),'N_employees':len(employees)})
pd.DataFrame(boot).to_csv(OUT/'reproduced_cluster_bootstrap.csv',index=False)

# Stage 11.2 robustness: employee-cluster bootstrap for H2/H3 associations.
# A fresh deterministic RNG is used so this analysis is reproducible independently
# of the H1 bootstrap above. Employee sampling retains all test windows for each
# sampled employee, preserving within-employee dependence.
rng_assoc=np.random.default_rng(20260914)
assoc_boot=[]
for name in ['Linear Regression','Random Forest','XGBoost']:
    div=np.abs(pred[('Manager',name)]-pred[('Benchmark',name)])
    deg=np.abs(yt-pred[('Manager',name)])-np.abs(yt-pred[('Benchmark',name)])
    for outcome,vals in [('abs_prediction_divergence',div),('error_degradation',deg)]:
        pr,_=pearsonr(DI,vals); sr,_=spearmanr(DI,vals)
        groups={e:np.where(emp==e)[0] for e in employees}
        bp=np.empty(5000); bs=np.empty(5000)
        for b in range(5000):
            sampled=rng_assoc.choice(employees,size=len(employees),replace=True)
            ids=np.concatenate([groups[e] for e in sampled])
            x=DI[ids]; v=vals[ids]
            bp[b]=np.corrcoef(x,v)[0,1]
            bs[b]=np.corrcoef(rankdata(x),rankdata(v))[0,1]
        assoc_boot.append({'Model':name,'Outcome':outcome,'Pearson_r':pr,
                           'Pearson_CI95_low':np.quantile(bp,.025),'Pearson_CI95_high':np.quantile(bp,.975),
                           'Spearman_rho':sr,'Spearman_CI95_low':np.quantile(bs,.025),'Spearman_CI95_high':np.quantile(bs,.975),
                           'N_windows':len(vals),'N_employees':len(employees),'Bootstrap_replicates':5000})
pd.DataFrame(assoc_boot).to_csv(OUT/'reproduced_cluster_bootstrap_H2_H3.csv',index=False)

# Stage 11.2 robustness: DI denominator sensitivity. Attendance has zero
# discrepancy variance, so excluding it yields only a positive linear rescaling.
nonzero=[f for f in features if sd[f] != 0]
DI5=Z[nonzero].mean(axis=1).to_numpy()
ratio=DI5/DI
valid=np.isfinite(ratio)
sens=[{'Check':'DI5_vs_DI6','N_dimensions_original':len(features),'N_dimensions_nonzero':len(nonzero),
       'Expected_scale_factor':len(features)/len(nonzero),
       'Max_abs_scaling_residual':np.max(np.abs(DI5-(len(features)/len(nonzero))*DI)),
       'Pearson_DI5_DI6':pearsonr(DI5,DI).statistic,'Spearman_DI5_DI6':spearmanr(DI5,DI).statistic}]
pd.DataFrame(sens).to_csv(OUT/'reproduced_discrepancy_index_sensitivity.csv',index=False)

print(pd.DataFrame(metric_rows).to_string(index=False))
print('\nTrain-only discrepancy associations:')
print(pd.DataFrame(assoc).to_string(index=False))
print('\nH1 Bootstrap:')
print(pd.DataFrame(boot).to_string(index=False))
print('\nH2/H3 cluster-bootstrap robustness:')
print(pd.DataFrame(assoc_boot).to_string(index=False))
print('\nDiscrepancy-index sensitivity:')
print(pd.DataFrame(sens).to_string(index=False))

# XGBoost SHAP global summaries (matched final test set)
try:
    import shap
    shap_rows=[]
    for view,X in [('Benchmark',XB),('Manager',XM)]:
        model=fitted[(view,'XGBoost')]
        sv=shap.TreeExplainer(model).shap_values(X.loc[test])
        sv=np.asarray(sv)
        for j,f in enumerate(features):
            shap_rows.append({'View':view,'Feature':f,'MeanAbsSHAP':np.abs(sv[:,j]).mean(),
                              'MeanSignedSHAP':sv[:,j].mean()})
    sg=pd.DataFrame(shap_rows)
    sg.to_csv(OUT/'reproduced_xgb_shap_global.csv',index=False)
    pv=sg.pivot(index='Feature',columns='View',values='MeanAbsSHAP').reset_index()
    pv['Delta_Manager_minus_Benchmark']=pv['Manager']-pv['Benchmark']
    pv['Abs_Delta']=pv['Delta_Manager_minus_Benchmark'].abs()
    pv=pv.sort_values('Abs_Delta',ascending=False)
    pv.to_csv(OUT/'reproduced_xgb_shap_comparison.csv',index=False)
except Exception as e:
    (OUT/'SHAP_REPRODUCTION_WARNING.txt').write_text('SHAP reproduction was skipped: '+repr(e)+'\n')

"""Reproducible benchmark. Run python train.py before streamlit run app.py."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NUM = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
FEATURES = ['Type'] + NUM

def features(frame):
    out = frame[FEATURES].copy()
    out['Temperature difference [K]'] = out[NUM[1]] - out[NUM[0]]
    return out

def evaluate(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    return dict(precision=float(precision_score(y,pred,zero_division=0)), recall=float(recall_score(y,pred,zero_division=0)), f1=float(f1_score(y,pred,zero_division=0)), average_precision=float(average_precision_score(y,scores)), accuracy=float(accuracy_score(y,pred)), confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())

def main():
    result = ROOT/'results'; result.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT/'data/ai4i2020.csv')
    X = features(df); y = df['Machine failure']
    tr, rest = train_test_split(df.index,test_size=.30,stratify=y,random_state=42)
    va, te = train_test_split(rest,test_size=.5,stratify=y.loc[rest],random_state=42)
    pre = ColumnTransformer([('type',OneHotEncoder(handle_unknown='ignore'),['Type']),('numeric','passthrough',NUM+['Temperature difference [K]'])])
    model = Pipeline([('preprocess',pre),('model',RandomForestClassifier(n_estimators=150,class_weight='balanced',min_samples_leaf=2,random_state=42,n_jobs=-1))])
    model.fit(X.loc[tr],y.loc[tr])
    scores = model.predict_proba(X.loc[va])[:,1]
    trials = [(float(t), evaluate(y.loc[va],scores,t)) for t in np.arange(.10,.81,.05)]
    threshold, validation = max(trials,key=lambda row:row[1]['f1'])
    test = evaluate(y.loc[te], model.predict_proba(X.loc[te])[:,1],threshold)
    dummy = DummyClassifier(strategy='most_frequent').fit(X.loc[tr],y.loc[tr])
    baseline = evaluate(y.loc[te],dummy.predict_proba(X.loc[te])[:,1],.5)
    normal = df.loc[tr][df.loc[tr,'Machine failure']==0]
    ranges = {c:{'low':float(normal[c].quantile(.05)),'high':float(normal[c].quantile(.95)),'median':float(normal[c].median()),'min':float(df.loc[tr,c].min()),'max':float(df.loc[tr,c].max())} for c in NUM}
    # Ordered holdout sensitivity check: no tuning against this holdout.
    from sklearn.base import clone
    ordered = clone(model).fit(X.iloc[:7000],y.iloc[:7000])
    ovs = ordered.predict_proba(X.iloc[7000:8500])[:,1]
    ot = max(np.arange(.10,.81,.05),key=lambda t:f1_score(y.iloc[7000:8500],ovs>=t,zero_division=0))
    ordered_test = evaluate(y.iloc[8500:],ordered.predict_proba(X.iloc[8500:])[:,1],float(ot))
    info = dict(rows=len(df), missing=int(df.isna().sum().sum()),duplicates=int(df.duplicated().sum()),class_counts={str(k):int(v) for k,v in y.value_counts().items()},split={'train':len(tr),'validation':len(va),'test':len(te)},threshold=threshold,validation=validation,test=test,baseline=baseline,ordered_test=ordered_test,ordered_threshold=float(ot),sha256=hashlib.sha256((ROOT/'data/ai4i2020.csv').read_bytes()).hexdigest())
    (result/'metrics.json').write_text(json.dumps(info,indent=2))
    joblib.dump({'model':model,'threshold':threshold,'ranges':ranges},result/'model.joblib',compress=3)
    df.loc[te,FEATURES+['Machine failure']].to_csv(result/'heldout_examples.csv',index=False)
    df.loc[te,FEATURES].head(10).to_csv(ROOT/'sample_input.csv',index=False)
    pd.DataFrame({'row':df.index,'split':['train' if i in set(tr) else 'validation' if i in set(va) else 'test' for i in df.index]}).to_csv(result/'split_indices.csv',index=False)
    fig,axs = plt.subplots(1,3,figsize=(14,4))
    y.value_counts().sort_index().plot.bar(ax=axs[0],color=['#14b8a6','#f97316'],rot=0)
    axs[0].set(title='Class imbalance',xlabel='Failure label',ylabel='Records')
    for label,color in [(0,'#14b8a6'),(1,'#f97316')]:
        axs[1].hist(df.loc[y==label,NUM[0]],bins=25,alpha=.65,label=str(label))
    axs[1].set(title='Air temperature',xlabel='Kelvin'); axs[1].legend(title='Failure')
    axs[2].scatter(df[NUM[2]],df[NUM[3]],c=y,cmap='coolwarm',s=5,alpha=.35)
    axs[2].set(title='Speed and torque',xlabel='RPM',ylabel='Nm')
    fig.tight_layout();fig.savefig(result/'eda.png',dpi=150);plt.close(fig)
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from train import FEATURES, NUM, features

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Machine Guardian',page_icon='⚙️',layout='wide')
st.markdown('''<style>.stApp {background: #08121f;} h1,h2,h3 {color:#dcecf4;} div[data-testid="stMetric"] {background:#132438;padding:18px;border-radius:12px;} .block-container {padding-top:2rem;}</style>''',unsafe_allow_html=True)
st.caption('MACHINE GUARDIAN / CONDITION ASSESSMENT LAB')
st.title('Machine readings. Clearer decisions.')
st.write('Inspect operating conditions with a trained classifier and transparent benchmark results.')
st.info('Educational prototype trained on synthetic data. Classifies recorded conditions; does not forecast failure time or certify machine safety.')
if not (ROOT/'results/model.joblib').exists():
    st.error('Model missing. Run: python train.py');st.stop()
@st.cache_resource
def load():
    return joblib.load(ROOT/'results/model.joblib')
@st.cache_data
def data():
    return pd.read_csv(ROOT/'data/ai4i2020.csv'),json.loads((ROOT/'results/metrics.json').read_text())
bundle=load();df,metrics=data();model=bundle['model'];threshold=bundle['threshold']

def validate(frame):
    missing=set(FEATURES)-set(frame.columns)
    if missing:raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    if frame.empty:raise ValueError('CSV has no rows.')
    if len(frame)>10000:raise ValueError('Please use at most 10,000 rows.')
    clean=frame[FEATURES].copy()
    clean['Type']=clean['Type'].astype(str).str.strip().str.upper()
    if not clean['Type'].isin(['L','M','H']).all():raise ValueError('Type must be L, M or H.')
    for c in NUM:clean[c]=pd.to_numeric(clean[c],errors='raise')
    if not np.isfinite(clean[NUM].to_numpy(dtype=float)).all():raise ValueError('Numeric values must be finite and non-missing.')
    if (clean[NUM[:4]]<=0).any().any() or (clean[NUM[4]]<0).any():raise ValueError('Temperatures, speed and torque must be positive; tool wear must be non-negative.')
    return clean

def predict(frame):
    clean=validate(frame)
    scores=model.predict_proba(features(clean))[:,1]
    out=clean.copy();out['Failure score']=scores;out['Predicted failure']=(scores>=threshold).astype(int)
    out['Outside training range']=False
    for c in NUM:
        r=bundle['ranges'][c]
        out['Outside training range'] |= (clean[c]<r['min']) | (clean[c]>r['max'])
    return out

predict_tab,batch_tab,eda_tab,eval_tab=st.tabs(['01 · Inspect','02 · Batch CSV','03 · Dataset','04 · Evaluation'])
with predict_tab:
    st.subheader('Inspect one operating snapshot')
    st.caption('Choose an example, or change values to explore model sensitivity. Examples are held-out records, not training rows.')
    examples=pd.read_csv(ROOT/'results/heldout_examples.csv')
    choice=st.selectbox('Starting example',['Normal-labelled example','Failure-labelled example'])
    row=examples[examples['Machine failure']==int(choice.startswith('Failure'))].iloc[0]
    with st.form('snapshot'):
        cols=st.columns(3)
        vals={'Type':cols[0].selectbox('Product quality type',['L','M','H'],index=['L','M','H'].index(row['Type']),key='type'+choice)}
        for i,c in enumerate(NUM):
            vals[c]=cols[(i+1)%3].number_input(c,min_value=0.0,value=float(row[c]),step=.1,key=c+choice)
        submitted=st.form_submit_button('Assess conditions',type='primary')
    if submitted:
        try:
            output=predict(pd.DataFrame([vals]));record=output.iloc[0]
            a,b,c=st.columns(3)
            a.metric('Classification','Failure flagged' if record['Predicted failure'] else 'No failure flagged')
            b.metric('Model score',f"{record['Failure score']:.3f}")
            c.metric('Decision threshold',f'{threshold:.2f}')
            st.caption('Score is uncalibrated model output, not a verified probability of real-world failure. No failure flagged does not guarantee safety.')
            if record['Outside training range']:st.warning('One or more values are outside observed training ranges. This prediction may be unreliable.')
            refs=[]
            for c in NUM:
                r=bundle['ranges'][c];v=vals[c]
                refs.append({'Reading':c,'Entered':v,'Normal reference P5':round(r['low'],2),'Normal reference P95':round(r['high'],2),'Reference status':'Outside central range' if v<r['low'] or v>r['high'] else 'Within central range'})
            st.dataframe(pd.DataFrame(refs),hide_index=True,width='stretch')
            st.caption('Reference range = 5th–95th percentiles of normal-labelled TRAINING data. These are not safety limits or causal explanations. Changing values is a model experiment, not a repair instruction.')
        except ValueError as e:st.error(str(e))
with batch_tab:
    st.subheader('Assess a batch of readings')
    st.download_button('Download CSV template',data=(ROOT/'sample_input.csv').read_bytes(),file_name='sample_input.csv',mime='text/csv')
    upload=st.file_uploader('CSV with the six input columns',type='csv')
    if upload is not None:
        try:
            result=predict(pd.read_csv(upload))
            st.dataframe(result,hide_index=True,width='stretch')
            st.download_button('Download predictions',result.to_csv(index=False),file_name='machine_guardian_predictions.csv',mime='text/csv')
        except (ValueError,TypeError,pd.errors.ParserError) as e:st.error('Please check your file: '+str(e))
with eda_tab:
    a,b,c=st.columns(3);a.metric('Records',f'{len(df):,}');b.metric('Failure records',metrics['class_counts']['1']);c.metric('Missing cells',metrics['missing'])
    st.image(str(ROOT/'results/eda.png'),width='stretch')
    st.dataframe(df[FEATURES+['Machine failure']].describe(include='all').fillna('').astype(str),width='stretch')
    st.markdown('Source: [UCI AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), CC BY 4.0. Synthetic data; 10,000 rows. No audio or vibration inputs.')
with eval_tab:
    st.subheader('Held-out benchmark')
    st.write('70% training / 15% validation / 15% test; seed 42. Threshold selected by validation F1, then frozen before test evaluation.')
    comparison=pd.DataFrame({k:metrics[k] for k in ['baseline','test','ordered_test']}).T.drop(columns=['confusion_matrix'])
    comparison.index=['Majority-class baseline','Stratified test','Ordered holdout sensitivity']
    st.dataframe(comparison.style.format('{:.3f}'),width='stretch')
    st.write('Confusion matrix — rows: actual; columns: predicted')
    st.dataframe(pd.DataFrame(metrics['test']['confusion_matrix'],index=['Actual normal','Actual failure'],columns=['Predicted normal','Predicted failure']))
    st.caption('Ordered sensitivity model trains on the first 70%, tunes on the next 15%, tests on the last 15%. Row order is not verified real-world time. Neither evaluation establishes performance at another factory.')
    st.warning('Dataset failures are generated using synthetic rules. Results demonstrate a benchmark workflow, not validated industrial predictive maintenance.')

# Machine Guardian
A working Streamlit machine-condition assessment MVP, trained on the UCI AI4I 2020 synthetic dataset. It classifies recorded operating conditions, not future time-to-failure.

## Quick start (Python 3.11 or 3.12 recommended)
Extract the ZIP, then open a terminal in the machine_guardian folder.

```bash
python -m venv .venv
```
Windows activation: `.venv\Scripts\activate`
macOS/Linux activation: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```
Open the localhost address displayed in the terminal. The dataset is included: no download or API key is needed for normal use. A trained model is included, but running train.py ensures compatibility with your installed environment. Only load the trusted model supplied with this project; joblib/pickle files can execute code.

## Demo (2 minutes)
1. Open Inspect. Run the normal-labelled held-out example.
2. Select the failure-labelled example and assess it.
3. Explain the score and fixed threshold. A failure-labelled sample need not be correctly classified; the app reports actual model predictions.
4. Compare entered readings with training-data reference ranges. These ranges are not safety limits.
5. Download the CSV template, upload it in Batch CSV, download results.
6. Show Evaluation, including missed failures and the ordered sensitivity result.

## Concepts
Data preprocessing; categorical encoding; EDA; temperature-difference feature engineering; supervised classification; validation-based threshold selection; imbalance handling; model evaluation; Streamlit deployment.

## Files
- app.py: interactive UI, validation and batch predictions
- train.py: reproducible training, EDA and evaluation
- data/ai4i2020.csv: original attributed dataset
- results/model.joblib: fitted preprocessing/model plus threshold and reference ranges
- results/metrics.json: observed evaluation results
- results/eda.png: EDA chart
- results/heldout_examples.csv: labelled demonstration records
- results/split_indices.csv: original zero-based row positions and split membership
- sample_input.csv: six-column upload template
- RND_REPORT.md: research findings, measured results, limitations, 60-minute plan

## Hosting
For Streamlit Community Cloud, push this folder to your own GitHub repository and choose app.py as the entrypoint. Include requirements.txt, data, results and train.py. Do not upload private data or secrets. Public hosting has not been performed for this package. Local Streamlit is the tested demo route. FastAPI and ngrok are not required for the MVP.

## Source and licence
AI4I 2020 Predictive Maintenance Dataset (2020), UCI Machine Learning Repository.
https://doi.org/10.24432/C5HS5C
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
Dataset licence: Creative Commons Attribution 4.0 https://creativecommons.org/licenses/by/4.0/
The included original CSV is unchanged. Derived charts, splits and predictions were produced for this educational project. No affiliation with UCI is implied.

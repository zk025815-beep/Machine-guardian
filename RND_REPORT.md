# Machine Guardian — R&D and implementation report

## Decision
Build a six-input Streamlit classifier for recorded machine conditions. Focus on a functional, explainable workflow, not a new algorithm. Real users would be maintenance technicians prioritizing inspection. Practical benefit is a hypothesis; no business savings or field reliability were measured.

## Research grounding
UCI describes AI4I as synthetic data reflecting industrial maintenance conditions, with 10,000 rows and six inputs. Its associated 2020 paper is S. Matzka, Explainable Artificial Intelligence for Predictive Maintenance Applications. Predictive maintenance and this benchmark are established; the project does not claim algorithmic novelty.

Source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
Model documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
Leakage guidance: https://scikit-learn.org/stable/common_pitfalls.html
Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
Deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Actual data audit
Downloaded the official ZIP and inspected the CSV locally: 10,000 rows, 14 columns, zero missing cells, zero exact duplicate rows. Normal class: 9,661 (96.61%); failure: 339 (3.39%). Failure class imbalance is the main evaluation concern. A majority-only model achieves high accuracy while detecting no failures.

Inputs: product quality Type L/M/H, air temperature K, process temperature K, rotational speed RPM, torque Nm, tool wear minutes. Type is product quality, not machine identity. There are no vibration, audio or explicit future-failure-horizon fields.

Excluded UDI, Product ID and all failure subtype labels (TWF/HDF/PWF/OSF/RNF) from model inputs. Failure subtype columns would reveal outcome information. Original dataset retained unchanged. Temperature difference computed solely from input readings. No synthetic oversampling, no automatic removal of extremes.

## Model experiment
Fixed seed 42. Stratified split 7,000 training / 1,500 validation / 1,500 test. One-hot encoding fitted within a pipeline. Random Forest: 150 trees, balanced class weights, minimum leaf size 2. No large tuning search. Validation F1 selected the score threshold from 0.10–0.80 in steps of 0.05. Selected threshold: 0.40. Test set was not used to choose it. Normal reference ranges were calculated exclusively from normal-labelled training rows.

| Metric | Majority baseline | Random Forest test |
|---|---:|---:|
| Accuracy | 96.60% | 98.80% |
| Failure precision | 0% | 88.37% |
| Failure recall | 0% | 74.51% |
| Failure F1 | 0 | 0.809 |
| Average precision | 0.034 | 0.853 |

Test confusion matrix: 1,444 true negatives, 5 false positives, 13 false negatives, 38 true positives. There were 51 failures in the test set, of which 38 were detected. Model scores are uncalibrated and should not be described as verified real-world failure probabilities.

## Sensitivity and limitations
A separately trained ordered split (first 70% train, next 15% validation, last 15% test) yielded precision 80.95%, recall 58.62%, F1 0.680 and average precision 0.657. Its independently validation-selected threshold was 0.45. This lower recall shows split sensitivity. Row order is not verified real time; the check is not a production forecast validation.

The data uses synthetic failure rules, so success can reflect learning those rules rather than transferable industrial behavior. Only one seed and one small held-out test set were used. No external factory dataset, calibrated probabilities, sensor integration, physical fault diagnosis or remaining-useful-life estimation. Normal-reference flags describe distribution differences, not why the model predicted failure. Human inspection remains separate from model output.

## Implemented workflow
Single snapshot input -> schema/unit validation -> derived feature -> saved preprocessing/model -> fixed decision threshold -> score and reference comparison. Batch CSV input adds downloadable predictions. EDA and evaluation tabs expose limitations and errors alongside performance. Inputs outside observed training numeric ranges generate warnings; impossible/non-finite inputs are rejected.

## Revised 60-minute classroom plan (PPT afterwards)
| Elapsed | Minutes | Work |
|---|---:|---|
| 00–08 | 8 | Confirm problem, inspect data, finalize topic and team names |
| 08–16 | 8 | Split, preprocessing and quick EDA |
| 16–28 | 12 | Train baseline/model; select threshold on validation |
| 28–34 | 6 | Freeze model; test metrics and error analysis |
| 34–49 | 15 | Streamlit inputs, prediction, EDA and evaluation UI |
| 49–56 | 7 | Test valid/invalid inputs, examples and batch path |
| 56–60 | 4 | Package, launch locally, rehearse demo |
| Total | 60 | Functional project; PPT is separate |

A public deployment attempt fits only if accounts/repository are already configured; it is not guaranteed within this window. Preserve a local demo. Freeze optional features at minute 49.

## PPT after the project (suggested 15–20 minutes)
1. Problem and target user.
2. Dataset, synthetic origin and class imbalance.
3. Pipeline and multiple course concepts.
4. Actual evaluation: precision, recall, confusion matrix, ordered sensitivity.
5. Live demo and practical limitations.
6. Future work: real sensor data, time-separated validation, calibrated scores and maintenance cost evaluation.

Every member should explain why labels are excluded from inputs, why accuracy is insufficient and why this is not a time-to-failure predictor.

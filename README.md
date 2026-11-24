# Fraud Detection Pipeline: PostgreSQL → Feature Engineering → LightGBM vs PyTorch → Docker API

End-to-end pipeline that flags fraudulent card payments in **590,540 real e-commerce
transactions** (IEEE-CIS / Vesta, 3.5% fraud): raw data in PostgreSQL, leakage-free
behaviour features in SQL, gradient boosting vs a hand-written PyTorch network, numbers
stress-tested (seeds, bootstrap, dollars), and the best model served as a Docker API.

**▶ Live demo: [fraud-risk-demo.onrender.com](https://fraud-risk-demo.onrender.com)** (free hosting: the first request after a pause can take up to a minute)

| | |
|---|---|
| **Best model** | LightGBM: **PR-AUC 0.515 ± 0.003** over 5 seeds (ROC-AUC 0.90), on a future month never used for training |
| **Business view** | Reviewing the riskiest 1% of payments catches **24% of fraud cases** at **85% precision** |
| **In dollars** | Ranking the same queue by expected loss catches **36% of fraud dollars** instead of 15% |
| **Stack** | PostgreSQL · SQL window functions · LightGBM · XGBoost · PyTorch · FastAPI · Docker · pytest |

<p align="center"><img src="reports/figures/demo_page.png" width="720" alt="Demo page: a real fraudulent transaction from the test month scored at 95%"></p>

## Results

Trained on months 1–4, early-stopped on month 5 (validation), reported on month 6 (test).

| Model | PR-AUC | 95% interval | Fraud caught @ 1% alerts | Alerts that are fraud |
|---|---|---|---|---|
| **LightGBM** | **0.521** | 0.48–0.56 | **24.3%** | **84.8%** |
| XGBoost | 0.500 | 0.46–0.54 | 24.3% | 84.5% |
| PyTorch MLP (embeddings) | 0.485 | 0.44–0.53 | 23.5% | 81.9% |
| Logistic regression (baseline) | 0.418 | 0.38–0.46 | 23.2% | 80.7% |
| Isolation Forest (no labels) | 0.043 | 0.04–0.05 | 0.4% | 1.2% |

Random guessing gives PR-AUC ≈ 0.035. The interval comes from resampling whole test days.

<details>
<summary><b>Full table</b>: ROC-AUC, and precision / recall / F1 at each model's max-F1 threshold</summary>

| Model | PR-AUC | ROC-AUC | Recall @ 1% alerts | Precision @ 1% alerts | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.4179 | 0.8499 | 23.15% | 80.65% | 50.36% | 37.99% | 0.4331 |
| XGBoost | 0.5004 | 0.8815 | 24.25% | 84.45% | 57.28% | 46.34% | 0.5123 |
| **LightGBM** | **0.5208** | **0.9008** | **24.34%** | **84.79%** | 56.33% | **49.58%** | **0.5274** |
| PyTorch MLP (embeddings) | 0.4847 | 0.8820 | 23.51% | 81.88% | 56.77% | 43.22% | 0.4908 |
| Isolation Forest (unsupervised) | 0.0433 | 0.5988 | 0.35% | 1.21% | 4.47% | 13.94% | 0.0677 |

The max-F1 threshold of each model is chosen on the validation month
([`reports/model_comparison.md`](reports/model_comparison.md)).

</details>

<p align="center"><img src="reports/figures/pr_curves.png" width="560" alt="Precision-recall curves"></p>

**Takeaways**
- **Gradient boosting wins on tabular data:** trees handle missing values, skewed amounts and
  high-cardinality IDs natively; the neural net needs log transforms, scaling and embeddings
  to get close.
- **One month cannot rank the top three precisely.** Their 95% intervals overlap; the gap to
  the logistic-regression baseline is real. Across 5 seeds LightGBM scores 0.515 ± 0.003, so
  the honest headline is 0.515, not the 0.521 of the best seed.
- **Case counts hide the money.** The riskiest 1% are mostly small frauds; see below.
- **Why PR-AUC, not accuracy:** with 3.5% fraud, "never fraud" is 96.5% accurate. PR-AUC
  rewards finding fraud without flooding analysts with false alarms.
- **Why a time split:** fraud drifts (weekly rate 1.9%–5.1%). A random split would let the
  model learn from the future.

### What the model catches in dollars

There is **$477k** of fraud in the test month. Same budget (review 1% of payments), two ways
to choose which ones ([`reports/alert_ranking.md`](reports/alert_ranking.md)):

| Review the top 1% by | Fraud cases caught | Reviewed cases that are fraud | Fraud dollars caught |
|---|---|---|---|
| fraud probability (API alert rule) | 24.3% | 84.8% | 15.1% ($71.9k) |
| **expected loss = probability × amount** | 14.6% | 50.9% | **35.9% ($171.3k)** |

Ranking by expected loss **more than doubles the money caught**, on the validation month too
(41.5% vs 17.7%). The cost: half of the reviewed payments are legitimate large purchases.
Which rule is right depends on the cost of a review, so the API returns both
`fraud_probability` and `expected_loss_usd`.

<details>
<summary><b>How much can the numbers be trusted?</b> Seeds, bootstrap, ensemble</summary>

[`src/robustness.py`](src/robustness.py) checks the headline instead of taking one run at face value.

- **Random seed.** LightGBM retrained with 5 seeds: test PR-AUC **0.5149 ± 0.0031**
  (0.5124–0.5189). Seed 42, used in the tables, happens to be the best run (0.5208).
- **Day-level bootstrap.** Resampling whole test days (fraud comes in campaigns, so single
  rows are not independent) gives 0.48–0.56 for LightGBM. Differences of 0.01–0.02 between
  the top models are within this noise.
- **Ensemble, decided on validation.** A rank average of LightGBM, XGBoost and the MLP is
  slightly better on validation (0.5924 vs 0.5868) and slightly worse on test (0.5190 vs
  0.5208). Both gaps are about the size of the seed noise, so the API keeps the single
  LightGBM: simpler to serve, no measurable loss.

Full table with dollar metrics per model: [`reports/robustness.md`](reports/robustness.md).

</details>

<details>
<summary><b>Which features are worth serving?</b> 45 served features vs ~430 raw columns</summary>

[`src/feature_ablation.py`](src/feature_ablation.py) retrains LightGBM with growing feature sets on the same split:

| Feature set | Features | PR-AUC | ROC-AUC | Recall @ 1% | Precision @ 1% |
|---|---|---|---|---|---|
| **base (served model)** | 45 | 0.5208 | 0.9008 | 24.34% | 84.79% |
| + C/D/M counters, time deltas, match flags | 66 | 0.5248 | 0.9013 | 24.28% | 84.56% |
| + V1–V339 (Vesta's anonymised features) | 405 | **0.5458** | **0.9038** | **25.50%** | **88.81%** |
| + identity (device, browser, network) | 445 | 0.5410 | 0.9033 | 25.27% | 88.03% |

- The V columns add **+0.025 PR-AUC**, about 8× the seed noise: a real gain. Identity adds nothing.
- The API stays on 45 features: an endpoint that needs 339 anonymised columns per request is
  hard to integrate and cannot be explained to analysts. If Vesta's feature service is
  available in production, the V set is the first upgrade.

</details>

<details>
<summary><b>Decision threshold</b>: why 0.748, chosen by alert budget</summary>

The threshold is **always chosen on the validation month** and applied unchanged to the test month.

- **API threshold: 0.748 (LightGBM)**, the score above which the top **1%** of validation
  payments fall. A fraud team can only review so many cases a day, so the threshold is set by
  capacity, not by a default like 0.5. On the test month it flags 0.99% of payments, catches
  24.18% of fraud, and 84.89% of alerts are real fraud (F1 0.3764). Stored in
  `models/api_config.json`; test results in `reports/api_threshold_test.json`.
- **Max-F1 threshold per model** flags more payments (higher recall, lower precision); it is the
  right choice when automatic actions (e.g. 3-D Secure step-up) are cheap.

</details>

<details>
<summary><b>Unsupervised baseline</b>: Isolation Forest</summary>

`src/train_iforest.py` fits an Isolation Forest **without labels** (`contamination` = fraud
rate in train). Random trees isolate outliers in few splits, so a short average path means an
anomaly. It shows how far "unusual" gets us when labels are late (chargebacks arrive weeks
later). Fraud that imitates normal behaviour is not an outlier, so it trails the supervised
models: PR-AUC 0.043, barely above random (0.035).

</details>

![Feature importance](reports/figures/feature_importance.png)

---

## Pipeline

```
Kaggle CSVs ──COPY──> PostgreSQL raw.*  ──SQL window functions──> features.transactions
                                                                           │
                                                           Parquet export  │
                                                                           ▼
                       ┌───────────── time split: train | valid | test ─────────────┐
                       │  LogReg baseline · XGBoost · LightGBM · PyTorch MLP        │
                       │  Isolation Forest (no labels) · feature-set ablation        │
                       └────────────────────────────┬───────────────────────────────┘
                                                    │ seeds · day bootstrap · $ metrics
                                                    │ best model
                                                    ▼
                                  FastAPI /predict  ──>  Docker image
```

### Feature engineering (`sql/04_features.sql`)
Every behavioural feature uses **only transactions before the current one**, exactly what a
real-time system knows when a payment arrives, so there is no target leakage.

| Feature | Fraud pattern it captures |
|---|---|
| `uid` = card1 + addr1 + card start day (`day − D1`) | Proxy customer ID (the dataset has none). The idea comes from the competition's 1st-place solution (Chris Deotte and Konstantin Yakovlev); this project computes the history features from it **from past transactions only**, in SQL |
| `uid_tx_last_1h`, `uid_tx_last_24h`, `card1_tx_last_24h` | **Velocity**: stolen cards are used fast, before they are blocked. 0 prior tx in 24 h → 2.4% fraud; 4+ → 7.7% |
| `amt_to_uid_mean`, `uid_prev_std_amt` | Amount unusual **for this customer** (raw amount barely separates classes) |
| `secs_since_prev_tx` | Bursts of rapid payments |
| `uid_same_amt_prev` | Repeated amounts look like subscriptions, i.e. legitimate |
| `amt_cents` | Odd cents (e.g. 49.817) mean currency conversion, i.e. a foreign merchant |
| `hour` | Night-time activity while the owner sleeps |
| `email_mismatch`, `has_identity` | Payer/recipient domain differ; device data present (7.9% vs 2.1% fraud) |

<details>
<summary><b>Data, models and serving in detail</b></summary>

**Data in PostgreSQL (`sql/`)**
- `01_create_raw_tables.sql`: 394 + 41 column raw tables, generated from the CSV headers by `src/generate_raw_ddl.py`
- `02_load_raw.sql`: bulk load with `COPY` (735k rows in ~70 s)
- `03_data_quality.sql`: class balance, duplicates, missing values, fraud rate over time and by segment ([output](reports/data_quality.txt))

**Models (`src/`)**
- `preprocess.py`: time split; category vocabulary and scaling fitted on train only
- `train_gbm.py`: logistic regression baseline, XGBoost and LightGBM with native categorical support and early stopping
- `nn_model.py` / `train_nn.py`: **PyTorch from scratch**: custom `Dataset`, `nn.Module` with one embedding table per categorical column, hand-written training loop, `BCEWithLogitsLoss` with `pos_weight` for class imbalance, AdamW, early stopping on validation PR-AUC
- `train_iforest.py`: unsupervised Isolation Forest baseline
- `evaluate.py`: PR-AUC, ROC-AUC, alert-budget metrics, threshold metrics, dollar metrics, day-level bootstrap
- `compare.py`, `robustness.py`, `feature_ablation.py`: comparison table, trust checks, feature-set ablation

**Serving (`src/api/`, `docker/`)**
- FastAPI: `POST /predict` returns the fraud probability, the expected loss and an alert flag; `GET /health`
- Demo page at `/` (plain HTML + JavaScript) with three real test-month transactions: legitimate, borderline and fraud
- Slim Docker image with only inference dependencies, running as a non-root user
- Deployed on Render from `render.yaml`: Render builds `docker/Dockerfile` straight from this repo
- `tests/test_parity.py` checks that the API reproduces offline model scores on real transactions, guarding against training/serving skew

</details>

<details>
<summary><b>Run it yourself</b></summary>

Requires Linux or WSL2, Docker and Python 3.12.

```bash
# 1. Data: accept the competition rules on Kaggle, then put the files in data/raw/
#    https://www.kaggle.com/competitions/ieee-fraud-detection/data

# 2. Database
cp .env.example .env
docker compose up -d
docker compose exec -T db psql -U fraud -d fraud -f /sql/01_create_raw_tables.sql
docker compose exec -T db psql -U fraud -d fraud -f /sql/02_load_raw.sql
docker compose exec -T db psql -U fraud -d fraud -f /sql/03_data_quality.sql
docker compose exec -T db psql -U fraud -d fraud -f /sql/04_features.sql

# 3. Python environment and training
bash scripts/setup_env.sh && source ~/.venvs/fraud/bin/activate
python -m src.build_dataset
python -m src.train_gbm
python -m src.train_nn
python -m src.train_iforest
python -m src.compare
python -m src.robustness        # seeds, bootstrap intervals, dollars, ensemble, alert ranking
python -m src.feature_ablation  # what the unused raw columns would add
python -m pytest tests

# 4. API in Docker
docker build -f docker/Dockerfile -t fraud-api .
docker run -p 8080:8080 fraud-api
# demo page: http://localhost:8080/   ·   API docs: http://localhost:8080/docs
```

Example request:

```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"amt": 117.0, "hour": 3, "product_cd": "C", "card4": "visa", "card6": "credit", "uid_tx_last_24h": 6}'
```

</details>

## Limitations and next steps
- **One test month:** the PR-AUC interval is about ±0.04, so small gaps between the top models
  are not conclusive; the dataset covers only six months.
- **Customer history comes in the request;** in production it would come from a feature store
  or the PostgreSQL feature layer.
- **The model is trained to find cases, not money;** ranking by expected loss is a post-hoc fix.
  Next: amount-weighted training and choosing the rule with the fraud team's real review cost.
- Next: probability calibration, and monitoring of drift and alert volume after deployment.

## Project structure

```
├── sql/                  # schema, bulk load, data-quality checks, feature layer
├── src/                  # dataset export, preprocessing, models, evaluation
│   └── api/              # FastAPI inference service
├── docker/Dockerfile     # inference image
├── docker-compose.yml    # PostgreSQL for local development
├── tests/                # metric, API and training/serving parity tests
├── models/               # serving artifacts (LightGBM, preprocessor, threshold)
├── reports/              # metrics, robustness, alert ranking, feature ablation, figures
└── scripts/setup_env.sh  # Python environment setup
```

Data: [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Vesta Corporation), used under the competition rules; raw data is not redistributed.

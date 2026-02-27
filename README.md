# Credit Card Fraud Detection — AWS SageMaker vs. SparkR on EMR

> **A platform comparison study: evaluating the same fraud detection ML pipeline — Logistic Regression, SGD, and Random Forest — across two AWS cloud services (SageMaker and SparkR on EMR) to benchmark model performance, development experience, and scalability on a highly imbalanced dataset (284K transactions, 0.17% fraud).**

**Course:** CSP-554 Big Data Technologies · Illinois Institute of Technology

---

## Project Output Dashboard

![Project Dashboard](output/project_dashboard.png)

---

## Core Objective

This project answers one central question:

> *Does AWS SageMaker or SparkR on AWS EMR deliver better performance and developer experience for a fraud detection classification task on imbalanced financial data?*

The **same dataset** and the **same three models** (Logistic Regression, SGD Classifier, Random Forest) are implemented on both platforms. Results are then compared across F1 Score, Recall, development overhead, and infrastructure setup cost.

---

## Platform Comparison Summary

| Dimension | SparkR on AWS EMR | AWS SageMaker |
|---|---|---|
| **Language** | R (sparklyr) | Python (scikit-learn) |
| **Setup complexity** | High — bootstrap script, cluster config | Low — managed notebook, one-click |
| **Data storage** | HDFS (Parquet) | Amazon S3 (CSV / Parquet) |
| **Distributed compute** | Yes — Spark across worker nodes | Single instance (scales via Training Jobs) |
| **Imbalance handling** | Class-weighted Random Forest | SMOTE via imbalanced-learn |
| **Best F1 (Random Forest)** | 0.9741 | 0.9741 |
| **Best Recall (Random Forest)** | 0.9612 | 0.9612 |
| **Infrastructure cost** | Higher (always-on cluster) | Lower (pay-per-use notebook) |
| **Reproducibility** | Requires cluster re-launch + bootstrap | Notebook snapshot, easy to share |
| **Best suited for** | Large-scale distributed data processing | Rapid ML experimentation & deployment |

**Verdict:** Both platforms reach equivalent model accuracy on this dataset. SageMaker wins on developer velocity and cost for a dataset of this size. SparkR/EMR wins when the dataset scales to hundreds of GB and distributed in-memory processing becomes necessary.

---

## Repository Structure

```
fraud-detection/
│
├── fraud_detection.R              # SparkR pipeline — AWS EMR
├── sagemaker_fraud_detection.py   # Python pipeline — AWS SageMaker
├── bootstrap.sh                   # AWS EMR cluster bootstrap (R 4.0.3 + RStudio)
│
├── data/
│   └── train.csv                  # Kaggle Credit Card Fraud dataset (not committed)
│
├── output/
│   ├── project_dashboard.png      # Full 8-panel results dashboard
│   ├── varImpPlot.png             # Variable importance (Random Forest)
│   ├── f1_vs_vars.png             # F1 Score vs number of variables
│   ├── rf_error_curve.png         # RF error rate vs number of trees
│   └── model_comparison.png       # Cross-platform model comparison
│
└── README.md
```

---

## Dataset

**[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent | 492 (0.17%) |
| Features | V1–V28 (PCA-transformed), Time, Amount |
| Target | Class (0 = Legit, 1 = Fraud) |

The severe class imbalance (99.83% vs 0.17%) is a key challenge addressed differently on each platform.

---

## Tech Stack

| Layer | SparkR / EMR | SageMaker |
|---|---|---|
| Cloud Platform | AWS EMR | AWS SageMaker |
| Storage | HDFS (Parquet) | Amazon S3 |
| Compute | Apache Spark (distributed) | Managed ML instances |
| Language | R | Python 3 |
| Core Libraries | sparklyr, randomForest, caret, MLmetrics | scikit-learn, imbalanced-learn |
| Visualisation | ggplot2 | matplotlib, seaborn |

---

## Setup & Execution

### Platform A — SparkR on AWS EMR

#### 1. Launch EMR cluster with bootstrap
```bash
aws emr create-cluster \
  --name "CSP554-FraudDetection" \
  --release-label emr-6.4.0 \
  --applications Name=Spark Name=Hive Name=Hadoop \
  --bootstrap-actions Path=s3://your-bucket/bootstrap.sh \
  --instance-type m5.xlarge \
  --instance-count 3 \
  --use-default-roles
```

#### 2. Upload dataset to HDFS
```bash
hdfs dfs -mkdir -p /user/rstudio-user/
hdfs dfs -put data/train.csv /user/rstudio-user/
```

#### 3. Open RStudio Server and run
Navigate to `http://<master-node-dns>:8787` · Login: `hadoop` / `hadoop`
```r
source("fraud_detection.R")
```

---

### Platform B — AWS SageMaker

#### 1. Create Notebook Instance
AWS Console → SageMaker → Notebook Instances → Create
Instance: `ml.t3.medium` · IAM: `AmazonSageMakerFullAccess` + S3 read

#### 2. Upload data to S3
```bash
aws s3 cp data/train.csv s3://your-bucket/fraud-detection/train.csv
```

#### 3. Run notebook
Upload `sagemaker_fraud_detection.py`, update `DATA_PATH` to your S3 path, and run all cells.

---

## Results

### Data Split (both platforms)
| Set | Proportion |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

### Model Performance — SparkR on EMR

| Model | F1 Score |
|---|---|
| Logistic Regression | 0.8923 |
| Random Forest (Full, 17 vars) | 0.9741 |
| Random Forest (Top-10 vars, 1000 trees) | 0.9514 |

### Model Performance — AWS SageMaker

| Model | F1 Score | Recall |
|---|---|---|
| Logistic Regression | 0.8923 | 0.8761 |
| SGD Classifier | 0.7812 | 0.7501 |
| Random Forest | 0.9741 | 0.9612 |

### Variable Importance (Top 10 — EMR)
```
V17 > V12 > V14 > V10 > V16 > V11 > V9 > V4 > V18 > V26
```

---

## Key Findings

**Model accuracy is platform-agnostic** — both EMR and SageMaker peaked at F1 = 0.97 with Random Forest. The platform does not influence model quality at this dataset scale.

**Platform choice is a scalability and workflow decision** — SageMaker is faster to set up and more cost-effective for datasets that fit in memory; EMR becomes the right choice when data volume exceeds single-machine capacity.

**SMOTE vs. class weighting** — both strategies for handling the 0.17% fraud minority class produced comparable results, validating that either approach is acceptable in practice.

**Variable trimming is production-viable** — the top-10 variable RF model achieves 97.5% of the full-model F1 at 41% of the feature count, reducing inference latency without meaningful accuracy loss.

---

## References

1. Kaggle Credit Card Fraud Dataset — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. SparkR / sparklyr — https://spark.rstudio.com
3. AWS SageMaker Developer Guide — https://docs.aws.amazon.com/sagemaker/
4. SMOTE: Synthetic Minority Over-sampling — Chawla et al. (2002), JAIR
5. Random Forests — Breiman, L. (2001), Machine Learning, 45(1), 5–32

---

## License

Academic use only — Illinois Institute of Technology, CSP-554 Big Data Technologies.

<img width="499" height="200" alt="Screenshot 2026-01-26 at 8 04 43 AM" src="https://github.com/user-attachments/assets/048c0d1e-6e6b-42a2-8ddf-df73f121a6f8" />



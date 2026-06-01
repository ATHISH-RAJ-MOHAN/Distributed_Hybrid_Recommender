# Scalable and Distributed Recommendation Engine using Apache Spark

A recommendation system built using collaborative filtering, gradient boosting, and hybrid modeling techniques on the Yelp dataset. The project also includes a Locality Sensitive Hashing (LSH) implementation for identifying similar businesses at scale.

## Features

* Locality Sensitive Hashing (LSH) using MinHash signatures
* Item-Based Collaborative Filtering with Pearson Similarity
* Model-Based Recommendation System using XGBoost
* Hybrid Recommendation System combining collaborative filtering and machine learning
* Distributed processing with PySpark

---

## Project Structure

```text
.
├── task1.py      # Locality Sensitive Hashing (LSH)
├── task2_1.py    # Item-Based Collaborative Filtering
├── task2_2.py    # Model-Based Recommendation System (XGBoost)
├── task2_3.py    # Hybrid Recommendation System
└── data/
    ├── yelp_train.csv
    ├── yelp_val.csv
    ├── yelp_test.csv
    ├── business.json
    ├── user.json
    ├── checkin.json
    ├── photo.json
    ├── tip.json
    └── similarity.csv
```

---

## Dataset

Download the Yelp Dataset:

https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset

Create a `data/` directory and place all dataset files inside it.

For the recommendation tasks, split the review data into:

* 60% Training Set (`yelp_train.csv`)
* 20% Validation Set (`yelp_val.csv`)
* 20% Test Set (`yelp_test.csv`)

Each split should contain:

```text
user_id,business_id,stars
```

The metadata files should remain unchanged:

* business.json
* user.json
* checkin.json
* photo.json
* tip.json

---

## Installation

Install the required dependencies:

```bash
pip install pyspark numpy pandas scikit-learn xgboost
```

---

## Running the Project

### Task 1 - Business Similarity using LSH

```bash
spark-submit task1.py data/yelp_train.csv output.csv
```

### Task 2.1 - Item-Based Collaborative Filtering

```bash
spark-submit task2_1.py data/yelp_train.csv data/yelp_val.csv output.csv
```

### Task 2.2 - Model-Based Recommendation System

```bash
spark-submit task2_2.py data/ data/yelp_val.csv output.csv
```

### Task 2.3 - Hybrid Recommendation System

```bash
spark-submit task2_3.py data/ data/yelp_val.csv output.csv
```

> Note: The `folder_path` argument in Tasks 2.2 and 2.3 refers to the `data/` directory containing all Yelp data files.

---

## Results

| Task     | Method        | Performance                         |
| -------- | ------------- | ----------------------------------- |
| Task 1   | LSH + MinHash | Precision = 1.0000, Recall = 1.0000 |
| Task 2.1 | Item-Based CF | RMSE = 1.0689                       |
| Task 2.2 | XGBoost Model | RMSE = 0.9783                       |
| Task 2.3 | Hybrid Model  | RMSE = 0.9812                       |

---

## Approach

### Locality Sensitive Hashing

* MinHash signature generation
* Banding technique for candidate pair generation
* Jaccard similarity verification

### Item-Based Collaborative Filtering

* Pearson similarity between businesses
* Similarity caching for faster predictions
* Cold-start handling using user and business averages

### Model-Based Recommendation

Feature engineering from Yelp metadata:

* Business ratings
* Review counts
* Categories
* Location information
* User statistics
* Check-ins
* Photos
* Tips
* Similarity-based features

Model:

```python
XGBRegressor(
    max_depth=6,
    learning_rate=0.07,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8
)
```

### Hybrid Recommendation

Weighted combination of collaborative filtering and XGBoost predictions:

```text
Final Score = α × CF + (1 − α) × XGBoost
```

with:

```text
α = 0.22
```

---

## Technologies

* Python
* PySpark
* XGBoost
* NumPy
* Pandas
* Scikit-Learn

---

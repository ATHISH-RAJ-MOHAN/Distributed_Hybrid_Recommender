import sys
import csv
import time
import json
import numpy as np
import pandas as pd
from pyspark import SparkContext
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# SAFE GETTER
def safe_get(value, default):
    return default if value is None else value

'''
# LOCAL RMSE
def compute_rmse(preds, truths):
    se = 0.0
    for p, t in zip(preds, truths):
        se += (p - t) ** 2
    return (se / len(preds)) ** 0.5
'''

# FEATURE BUILDER FOR XGB
def build_features(uid, bid, bd, ud, cd, pd, td, sd):
    b = bd.get(bid, {})
    b_stars = b.get("stars", 3.5)
    b_review = b.get("review_count", 0)
    b_cat = b.get("categories_count", 0)
    b_lat = b.get("latitude", 0.0)
    b_lon = b.get("longitude", 0.0)

    u = ud.get(uid, {})
    u_review = u.get("review_count", 0)
    u_avg = u.get("average_stars", 3.5)
    u_useful = u.get("useful", 0)
    u_fans = u.get("fans", 0)

    checkins = cd.get(bid, 0)
    photos = pd.get(bid, 0)
    tip_count, _ = td.get(bid, (0, 0))
    sim_count, sim_sum = sd.get(bid, (0, 0))

    taste_diff = abs(u_avg - b_stars)

    return [
        b_stars, b_review, b_cat, b_lat, b_lon,
        checkins, photos, tip_count, sim_count, sim_sum,
        u_review, u_avg, u_useful, u_fans,
        taste_diff
    ]

# ITEM-BASED CF
def pearson_similarity(item1, item2, item_users, item_user_rating):
    users1 = set(item_users.get(item1, []))
    users2 = set(item_users.get(item2, []))
    co_users = users1.intersection(users2)

    if len(co_users) == 0:
        return 0.0

    r1, r2 = [], []
    for u in co_users:
        r1.append(item_user_rating[(item1, u)])
        r2.append(item_user_rating[(item2, u)])

    mean1 = sum(r1) / len(r1)
    mean2 = sum(r2) / len(r2)

    num = sum((a - mean1) * (b - mean2) for a, b in zip(r1, r2))
    den1 = sum((a - mean1) ** 2 for a in r1) ** 0.5
    den2 = sum((b - mean2) ** 2 for b in r2) ** 0.5

    if den1 == 0 or den2 == 0:
        return 0.0

    return num / (den1 * den2)

similarity_cache = {}

def get_similarity(i, j, item_users, item_user_rating):
    if (i, j) in similarity_cache:
        return similarity_cache[(i, j)]
    if (j, i) in similarity_cache:
        return similarity_cache[(j, i)]

    sim = pearson_similarity(i, j, item_users, item_user_rating)
    similarity_cache[(i, j)] = sim
    return sim

def predict_cf(user, item, user_items, item_users, item_user_rating, business_avg, user_avg, k=30):
    if user not in user_items:
        return 0.5 * business_avg.get(item, 3.0) + 0.5 * 3.0

    rated_items = user_items[user]
    sims = []

    for other_item in rated_items:
        if other_item == item:
            continue

        sim = get_similarity(item, other_item, item_users, item_user_rating)
        if sim == 0:
            continue

        rating = item_user_rating[(other_item, user)]
        sims.append((rating, sim))

    if len(sims) == 0:
        return 0.5 * business_avg.get(item, 3.0) + 0.5 * user_avg.get(user, 3.0)

    sims.sort(key=lambda x: x[1], reverse=True)
    sims = sims[:k]

    num = sum(r * s for r, s in sims)
    den = sum(abs(s) for _, s in sims)

    item_cf = None if den == 0 else num / den

    b_avg = business_avg.get(item, 3.0)
    u_avg = user_avg.get(user, 3.0)

    if item_cf is None:
        return 0.5 * b_avg + 0.5 * u_avg

    return 0.1 * item_cf + 0.5 * b_avg + 0.4 * u_avg

# MAIN
if __name__ == "__main__":
    start = time.time()

    folder = sys.argv[1]
    train_file = folder + "yelp_train.csv"
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    sc = SparkContext("local[*]", "task2_3")
    sc.setLogLevel("ERROR")

    # LOAD TRAINING DATA
    rdd = sc.textFile(train_file)
    header = rdd.first()

    ratings_rdd = (
        rdd.filter(lambda x: x != header)
           .map(lambda x: x.split(','))
           .map(lambda x: (x[0], x[1], float(x[2])))
    )

    user_items = (
        ratings_rdd.map(lambda x: (x[0], x[1]))
                   .groupByKey()
                   .mapValues(lambda items: list(set(items)))
                   .collectAsMap()
    )

    item_users = (
        ratings_rdd.map(lambda x: (x[1], x[0]))
                   .groupByKey()
                   .mapValues(lambda users: list(set(users)))
                   .collectAsMap()
    )

    item_user_rating = (
        ratings_rdd.map(lambda x: ((x[1], x[0]), x[2]))
                   .collectAsMap()
    )

    business_avg = (
        ratings_rdd.map(lambda x: (x[1], x[2]))
                   .groupByKey()
                   .mapValues(lambda vals: sum(vals) / len(vals))
                   .collectAsMap()
    )

    user_avg = (
        ratings_rdd.map(lambda x: (x[0], x[2]))
                   .groupByKey()
                   .mapValues(lambda vals: sum(vals) / len(vals))
                   .collectAsMap()
    )

    # LOAD SIDE DATA FOR XGB
    base = folder

    business_rdd = (
        sc.textFile(base + "business.json")
          .map(lambda line: json.loads(line))
          .map(lambda b: (
              b["business_id"],
              {
                  "stars": safe_get(b.get("stars"), 3.5),
                  "review_count": safe_get(b.get("review_count"), 0),
                  "categories_count": len(b.get("categories", "").split(",")) if b.get("categories") else 0,
                  "latitude": safe_get(b.get("latitude"), 0.0),
                  "longitude": safe_get(b.get("longitude"), 0.0)
              }
          ))
    ).collectAsMap()

    user_rdd = (
        sc.textFile(base + "user.json")
          .map(lambda line: json.loads(line))
          .map(lambda u: (
              u["user_id"],
              {
                  "review_count": safe_get(u.get("review_count"), 0),
                  "average_stars": safe_get(u.get("average_stars"), 3.5),
                  "useful": safe_get(u.get("useful"), 0),
                  "fans": safe_get(u.get("fans"), 0),
              }
          ))
    ).collectAsMap()

    checkin_rdd = (
        sc.textFile(base + "checkin.json")
          .map(lambda line: json.loads(line))
          .map(lambda c: (c["business_id"], sum(c.get("time", {}).values())))
    ).collectAsMap()

    photo_rdd = (
        sc.textFile(base + "photo.json")
          .map(lambda line: json.loads(line))
          .map(lambda p: (p["business_id"], 1))
          .reduceByKey(lambda a, b: a + b)
    ).collectAsMap()

    tip_rdd = (
        sc.textFile(base + "tip.json")
          .map(lambda line: json.loads(line))
          .map(lambda t: (t["business_id"], (1, safe_get(t.get("likes"), 0))))
          .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    ).collectAsMap()

    sim_rdd = (
        sc.textFile(base + "similarity.csv")
          .map(lambda x: x.split(","))
          .map(lambda x: (x[0], float(x[2])))
          .groupByKey()
          .mapValues(lambda vals: (len(vals), sum(vals)))
    ).collectAsMap()

    # TRAIN XGB
    train_pairs = ratings_rdd.collect()

    train_feats = []
    train_labels = []

    for uid, bid, rating in train_pairs:
        feats = build_features(uid, bid, business_rdd, user_rdd, checkin_rdd, photo_rdd, tip_rdd, sim_rdd)
        train_feats.append(feats)
        train_labels.append(rating)

    train_feats = np.array(train_feats)
    train_labels = np.array(train_labels)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feats)

    model = xgb.XGBRegressor(
        max_depth=6,
        learning_rate=0.07,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:linear",
        n_jobs=4
    )
    model.fit(train_scaled, train_labels)

    # LOAD TEST DATA
    test_rdd = sc.textFile(test_file)
    test_header = test_rdd.first()

    test_pairs = (
        test_rdd.filter(lambda x: x != test_header)
                .map(lambda x: x.split(","))
                .map(lambda x: (x[0], x[1]))
                .collect()
    )

    # PREDICT CF + XGB + HYBRID
    alpha = 0.22 

    final_preds = []
    output_rows = []

    for uid, bid in test_pairs:
        cf_pred = predict_cf(uid, bid, user_items, item_users, item_user_rating, business_avg, user_avg)

        feats = build_features(uid, bid, business_rdd, user_rdd, checkin_rdd, photo_rdd, tip_rdd, sim_rdd)
        xgb_pred = model.predict(scaler.transform([feats]))[0]

        hybrid = alpha * cf_pred + (1 - alpha) * xgb_pred

        final_preds.append(hybrid)
        output_rows.append((uid, bid, hybrid))


    # LOCAL RMSE CHECK
    '''
    if "val" in test_file:
        true_vals = []
        with open(test_file, "r") as f:
            next(f)
            for line in f:
                u, b, r = line.strip().split(",")
                true_vals.append(float(r))

        rmse = compute_rmse(final_preds, true_vals)
        print("LOCAL RMSE =", rmse)
    '''

    # WRITE OUTPUT
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        for row in output_rows:
            writer.writerow([row[0],row[1],row[2]])

    print("Duration =", time.time() - start)
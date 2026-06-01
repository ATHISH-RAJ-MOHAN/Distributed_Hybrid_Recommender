import sys
import time
import json
import numpy as np
import pandas as pd
from pyspark import SparkContext
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Safe getter
def safe_get(value, default):
    return default if value is None else value

# Feature builder (pure Python)
def build_features(uid, bid, bd, ud, cd, pd, td, sd):
    # Business features
    b = bd.get(bid, {})
    b_stars = b.get("stars", 3.5)
    b_review = b.get("review_count", 0)
    b_cat = b.get("categories_count", 0)
    b_lat = b.get("latitude", 0.0)
    b_lon = b.get("longitude", 0.0)

    # User features
    u = ud.get(uid, {})
    u_review = u.get("review_count", 0)
    u_avg = u.get("average_stars", 3.5)
    u_useful = u.get("useful", 0)
    u_fans = u.get("fans", 0)

    # Checkin / photo / tip / similarity
    checkins = cd.get(bid, 0)
    photos = pd.get(bid, 0)
    tip_count, _ = td.get(bid, (0, 0))
    sim_count, sim_sum = sd.get(bid, (0, 0))

    # Interaction feature
    taste_diff = abs(u_avg - b_stars)

    return [
        b_stars, b_review, b_cat, b_lat, b_lon,
        checkins, photos, tip_count, sim_count, sim_sum,
        u_review, u_avg, u_useful, u_fans,
        taste_diff
    ]

if __name__ == "__main__":
    start = time.time()

    folder = sys.argv[1]
    train_file = folder + "yelp_train.csv"     
    test_file  = sys.argv[2]                   
    output_file = sys.argv[3]

    sc = SparkContext("local[*]", "task2_2_xgb")
    sc.setLogLevel("ERROR")

    base = "../resource/asnlib/public/"
    business_path   = base + "business.json"
    user_path       = base + "user.json"
    checkin_path    = base + "checkin.json"
    photo_path      = base + "photo.json"
    tip_path        = base + "tip.json"
    similarity_path = base + "similarity.csv"

    # Load train/test pairs
    train_rdd = sc.textFile(train_file)
    h1 = train_rdd.first()
    train_pairs = (
        train_rdd.filter(lambda x: x != h1)
                 .map(lambda x: x.split(","))
                 .map(lambda x: (x[0], x[1], float(x[2])))
    )

    test_rdd = sc.textFile(test_file)
    h2 = test_rdd.first()
    test_pairs = (
        test_rdd.filter(lambda x: x != h2)
                .map(lambda x: x.split(","))
                .map(lambda x: (x[0], x[1]))
    )

    # Load side data with Spark
    business_rdd = (
        sc.textFile(business_path)
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
    ).cache()

    user_rdd = (
        sc.textFile(user_path)
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
    ).cache()

    checkin_rdd = (
        sc.textFile(checkin_path)
          .map(lambda line: json.loads(line))
          .map(lambda c: (c["business_id"], sum(c.get("time", {}).values())))
    ).cache()

    photo_rdd = (
        sc.textFile(photo_path)
          .map(lambda line: json.loads(line))
          .map(lambda p: (p["business_id"], 1))
          .reduceByKey(lambda a, b: a + b)
    ).cache()

    tip_rdd = (
        sc.textFile(tip_path)
          .map(lambda line: json.loads(line))
          .map(lambda t: (t["business_id"], (1, safe_get(t.get("likes"), 0))))
          .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    ).cache()

    similarity_rdd = (
        sc.textFile(similarity_path)
          .map(lambda x: x.split(","))
          .map(lambda x: (x[0], float(x[2])))
          .groupByKey()
          .mapValues(lambda vals: (len(vals), sum(vals)))
    ).cache()

    # Collect side data once
    business_dict = business_rdd.collectAsMap()
    user_dict = user_rdd.collectAsMap()
    checkin_dict = checkin_rdd.collectAsMap()
    photo_dict = photo_rdd.collectAsMap()
    tip_dict = tip_rdd.collectAsMap()
    sim_dict = similarity_rdd.collectAsMap()

    # Broadcast dictionaries
    business_bc = sc.broadcast(business_dict)
    user_bc = sc.broadcast(user_dict)
    checkin_bc = sc.broadcast(checkin_dict)
    photo_bc = sc.broadcast(photo_dict)
    tip_bc = sc.broadcast(tip_dict)
    sim_bc = sc.broadcast(sim_dict)

    # Build train features via mapPartitions
    def train_partition(iterable):
        bd = business_bc.value
        ud = user_bc.value
        cd = checkin_bc.value
        pd = photo_bc.value
        td = tip_bc.value
        sd = sim_bc.value
        for uid, bid, rating in iterable:
            feats = build_features(uid, bid, bd, ud, cd, pd, td, sd)
            yield (feats, rating)

    train_feat_rdd = train_pairs.mapPartitions(train_partition)
    train_collected = train_feat_rdd.collect()
    train_data, train_labels = zip(*train_collected)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    # Scale + train XGBoost
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)

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

    # Build test features via mapPartitions
    def test_partition(iterable):
        bd = business_bc.value
        ud = user_bc.value
        cd = checkin_bc.value
        pd = photo_bc.value
        td = tip_bc.value
        sd = sim_bc.value
        for uid, bid in iterable:
            feats = build_features(uid, bid, bd, ud, cd, pd, td, sd)
            yield (uid, bid, feats)

    test_feat_rdd = test_pairs.mapPartitions(test_partition)
    test_collected = test_feat_rdd.collect()

    test_users = [x[0] for x in test_collected]
    test_businesses = [x[1] for x in test_collected]
    test_data = np.array([x[2] for x in test_collected])

    test_scaled = scaler.transform(test_data)
    preds = model.predict(test_scaled)

    # Output
    df_out = pd.DataFrame({
        "user_id": test_users,
        " business_id": test_businesses,
        " prediction": preds
    })
    df_out.to_csv(output_file, index=False)

    print("Duration:", time.time() - start)
    sc.stop()
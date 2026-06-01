import sys
import csv
import time
from pyspark import SparkContext

start = time.time()

if __name__ == "__main__":
    train_filepath = sys.argv[1]
    test_filepath = sys.argv[2]
    output_filepath = sys.argv[3]

    sc = SparkContext('local[*]', 'item_cf')
    sc.setLogLevel("ERROR")

    # LOAD TRAINING DATA
    rdd = sc.textFile(train_filepath)
    header = rdd.first()

    ratings_rdd = (
        rdd.filter(lambda x: x != header)
           .map(lambda x: x.split(','))
           .filter(lambda x: len(x) >= 3)
           .map(lambda x: (x[0], x[1], float(x[2])))
    )

    # user -> items
    user_items = (
        ratings_rdd.map(lambda x: (x[0], x[1]))
                   .groupByKey()
                   .mapValues(lambda items: list(set(items)))
                   .collectAsMap()
    )

    # item -> users
    item_users = (
        ratings_rdd.map(lambda x: (x[1], x[0]))
                   .groupByKey()
                   .mapValues(lambda users: list(set(users)))
                   .collectAsMap()
    )

    # (item, user) -> rating
    item_user_rating = (
        ratings_rdd.map(lambda x: ((x[1], x[0]), x[2]))
                   .collectAsMap()
    )

    # business average
    business_avg = (
        ratings_rdd.map(lambda x: (x[1], x[2]))
                   .groupByKey()
                   .mapValues(lambda vals: sum(vals) / len(vals))
                   .collectAsMap()
    )

    # user average
    user_avg = (
        ratings_rdd.map(lambda x: (x[0], x[2]))
                   .groupByKey()
                   .mapValues(lambda vals: sum(vals) / len(vals))
                   .collectAsMap()
    )

    # PEARSON SIMILARITY
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

    # SIMILARITY CACHE
    similarity_cache = {}

    def get_similarity(i, j):
        if (i, j) in similarity_cache:
            return similarity_cache[(i, j)]
        if (j, i) in similarity_cache:
            return similarity_cache[(j, i)]

        sim = pearson_similarity(i, j, item_users, item_user_rating)
        similarity_cache[(i, j)] = sim
        return sim

    # HYBRID PREDICTION
    def predict_rating(user, item, user_items, item_users, item_user_rating, k=30):

        if user not in user_items:
            return 0.5 * business_avg.get(item, 3.0) + 0.5 * 3.0

        rated_items = user_items[user]
        sims = []

        for other_item in rated_items:
            if other_item == item:
                continue

            sim = get_similarity(item, other_item)
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

    # LOAD TEST FILE
    test_rdd = sc.textFile(test_filepath)
    test_header = test_rdd.first()

    test_pairs = (
        test_rdd.filter(lambda x: x != test_header)
                .map(lambda x: x.split(","))
                .filter(lambda x: len(x) >= 2)
                .map(lambda x: (x[0], x[1]))
                .collect()
    )

    # GENERATE PREDICTIONS
    predictions = []

    for user, item in test_pairs:
        pred = predict_rating(user, item, user_items, item_users, item_user_rating)
        predictions.append((user, item, pred))

    # WRITE OUTPUT
    with open(output_filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        for row in predictions:
            writer.writerow([row[0],row[1],row[2]])

    end = time.time()
    print("Duration =", end - start)
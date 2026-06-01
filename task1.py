import sys
import random
import time

from pyspark import SparkContext


def parse_line(line):
    # user_id, business_id, stars
    
    parts = line.strip().split(',')
    return parts[0], parts[1] #(user_id, business_id)

def jaccard(s1,s2):
    inter = len(s1 & s2)
    if inter == 0:
        return 0.0
    union = len(s1 | s2)
    
    return inter / union

def main():
    start = time.time()
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('ERROR')
    
    # 1. Load and clean
    raw = sc.textFile(input_file)
    header = raw.first()
    data = raw.filter(lambda x:x!=header).map(parse_line) #(user_id, business_id)
    
    # 2. business -> set(users)
    business_users = (
    data.map(lambda x:(x[1], x[0])) # (business, user)
        .groupByKey()
        .mapValues(lambda u: set(u)) # (business, set(user))
        .cache()
    )
    
    # 3. user -> index
    all_users = business_users.flatMap(lambda x:x[1]).distinct()
    user_index = all_users.zipWithIndex().collectAsMap()
    num_users = len(user_index)
    
    # 4. Minhash parameters
    num_hashes = 60
    bands = 20
    rows_per_band = num_hashes // bands
    m = num_users
    
    random.seed(42)
    hash_params = []
    
    for _ in range(num_hashes):
        a = random.randint(1, m-1)
        b = random.randint(0, m-1)
        hash_params.append((a, b))
        
       
    def minhash_signature(users_set):
        # users_set: set of user_ids
        indices = [user_index[u] for u in users_set]
        sig = []
        for (a,b) in hash_params:
            min_val = min(((a * idx + b) % m) for idx in indices)
            sig.append(min_val)
        return sig
    
    # 5. signature matrix: -> business -> signatiure list
    business_sig = business_users.mapValues(minhash_signature).cache()
    
    # 6. LSH banding
    def bands_for_business(business_sig_pair):
        bid, sig = business_sig_pair
        res = []
        for band in range(bands):
            start = band * rows_per_band
            end = start + rows_per_band
            band_key = (band, tuple(sig[start:end]))
            res.append((band_key, bid))
            
        return res
    
    band_buckets = (
    business_sig.flatMap(bands_for_business).groupByKey().mapValues(list)
    )
    
    # 7. Generate candidate pairs from buckets
    def gen_pairs(business_list):
        business_list = sorted(set(business_list))
        pairs = []
        n = len(business_list)
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((business_list[i], business_list[j]))
        return pairs
   
    candidate_pairs = (
    band_buckets.filter(lambda x:len(x[1]) > 1)
    .flatMap(lambda x:gen_pairs(x[1]))
    .distinct())
    
    
    # 8. Prepare for Jaccard: business -> users map
    business_users_map = business_users.collectAsMap()
    
    def compute_jaccard(pair):
        b1, b2 = pair
        s1 = business_users_map[b1]
        s2 = business_users_map[b2]
        sim = jaccard(s1, s2)
        
        return (b1, b2, sim)
    
    sims = candidate_pairs.map(compute_jaccard).filter(lambda x: x[2] >= 0.5)
    
    # 9. Sort and Format output
    result = (sims.map(lambda x:(x[0], x[1], x[2]))
             .sortBy(lambda x:(x[0], x[1]))
             .collect()
             )
    
    with open(output_file, 'w') as f:
        f.write("business_id_1, business_id_2, similarity\n")
        for b1, b2, sim in result:
            f.write(f"{b1},{b2},{sim}\n")
            
    '''# 10 - Precision and Recall
    gt_raw = sc.textFile("test_1_out.csv")
    gt_header = gt_raw.first()

    gt_pairs = (
        gt_raw.filter(lambda x: x != gt_header)
              .map(lambda x: x.split(','))
              .map(lambda x: (x[0], x[1]))
              .collect()
    )

    gt_set = set(gt_pairs)
    pred_set = set((b1, b2) for b1, b2, sim in result)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("Precision:", precision)
    print("Recall:", recall)'''

    
    end = time.time()
    runtime = end - start
    #print("Runtime:", runtime)

    sc.stop()
    
if __name__ == '__main__':
    main()
    
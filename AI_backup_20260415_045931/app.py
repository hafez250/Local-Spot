# =============================================================
# app.py - Flask API for Hybrid Recommender System
# Content-Based Filter + LightGBM
# =============================================================

from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import random
from sklearn.preprocessing import LabelEncoder
import hashlib

# =============================================================
# LOAD ALL FILES ON STARTUP
# =============================================================
print("Loading model files...")

with open("product_lookup.pkl", "rb") as f:
    product_lookup = pickle.load(f)

with open("user_profiles.pkl", "rb") as f:
    user_profiles = pickle.load(f)

with open("user_subcats.pkl", "rb") as f:
    user_subcats = pickle.load(f)

with open("features.json", "r") as f:
    feature_cols = json.load(f)

lgb_model = lgb.Booster(model_file="lightgbm_model.txt")

products_df = pd.read_csv("products_clean.csv")

# =============================================================
# REBUILD ENCODERS
# =============================================================
price_min = products_df["price"].min()
price_max = products_df["price"].max()
products_df["price_normalized"] = (products_df["price"] - price_min) / (price_max - price_min)

cat_enc = LabelEncoder()
subcat_enc = LabelEncoder()
products_df["category_encoded"] = cat_enc.fit_transform(products_df["category"])
products_df["subcategory_encoded"] = subcat_enc.fit_transform(products_df["subcategory"])

# Updated product lookup
product_lookup_updated = {}
for _, row in products_df.iterrows():
    product_lookup_updated[int(row["product_id"])] = {
        "name": row["name"],
        "category": row["category"],
        "subcategory": row["subcategory"],
        "price": row["price"],
        "category_encoded": int(row["category_encoded"]),
        "subcategory_encoded": int(row["subcategory_encoded"]),
        "price_normalized": float(row["price_normalized"])
    }

all_subcats = products_df["subcategory"].unique().tolist()
known_products = set(products_df["product_id"].tolist())

print("All models loaded ✅")

# =============================================================
# STEP 1: CONTENT-BASED FILTER → 50 CANDIDATES
# =============================================================
def _stable_seed(user_id, events):
    parts = []
    for e in events:
        pid = e.get("product_id")
        if pid is None:
            continue
        et = e.get("event_type", "purchase")
        parts.append(f"{pid}:{et}")

    seed_src = f"{user_id}|" + "|".join(parts)
    digest = hashlib.sha256(seed_src.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def get_candidates(events, user_id=0, topk=50):
    # استخرج الـ product IDs من الـ events
    weights = {"purchase": 3, "add_to_cart": 2, "view": 1}
    history_set = set()
    subcat_counts = {}

    for e in events:
        pid = e.get("product_id")
        if pid is None or pid not in product_lookup_updated:
            continue
        event_type = e.get("event_type", "purchase")
        weight = weights.get(event_type, 1)
        history_set.add(pid)

        subcat = product_lookup_updated[pid]["subcategory"]
        subcat_counts[subcat] = subcat_counts.get(subcat, 0) + weight

    rng = random.Random(_stable_seed(user_id, events))

    if not history_set:
        # Cold start — خد من كل الـ products عشوائي
        candidates = rng.sample(list(product_lookup_updated.keys()), min(topk, len(product_lookup_updated)))
        return candidates

    # اعرف الـ user بيحب إيه من الـ history
    from collections import Counter
    top_subcats = [s for s, _ in Counter(subcat_counts).most_common(3)]
    other_subcats = [s for s in all_subcats if s not in top_subcats]

    # 70% من المفضل + 30% تنوع
    n_primary = int(topk * 0.70)
    n_secondary = topk - n_primary

    primary_products = [
        pid for pid, p in product_lookup_updated.items()
        if p["subcategory"] in top_subcats and pid not in history_set
    ]

    secondary_products = [
        pid for pid, p in product_lookup_updated.items()
        if p["subcategory"] in other_subcats and pid not in history_set
    ]

    rng.shuffle(primary_products)
    rng.shuffle(secondary_products)

    candidates = primary_products[:n_primary] + secondary_products[:n_secondary]

    # Fallback لو مش كفاية
    if len(candidates) < topk:
        remaining = [p for p in product_lookup_updated.keys() if p not in candidates and p not in history_set]
        rng.shuffle(remaining)
        candidates += remaining[:topk - len(candidates)]

    return candidates[:topk]

# =============================================================
# STEP 2: LIGHTGBM → RANK CANDIDATES → TOP N
# =============================================================
def rank_with_lgb(candidates, user_id, limit=5):
    user_row = user_profiles[user_profiles["user_id"] == user_id]

    if len(user_row) == 0:
        fav_cat = 0
        fav_subcat = 0
        avg_price = float(products_df["price"].mean())
        purchase_count = 1
    else:
        fav_cat = int(user_row["fav_category"].iloc[0])
        fav_subcat = int(user_row["fav_subcategory"].iloc[0])
        avg_price = float(user_row["avg_price"].iloc[0])
        purchase_count = int(user_row["purchase_count"].iloc[0])

    rows = []
    for pid in candidates:
        if pid in product_lookup_updated:
            p = product_lookup_updated[pid]
            rows.append({
                "product_id": pid,
                "category_encoded": p["category_encoded"],
                "subcategory_encoded": p["subcategory_encoded"],
                "price": p["price"],
                "price_normalized": p["price_normalized"],
                "fav_category": fav_cat,
                "fav_subcategory": fav_subcat,
                "avg_price": avg_price,
                "purchase_count": purchase_count
            })

    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["score"] = lgb_model.predict(df[feature_cols])
    df["subcategory"] = df["product_id"].map(lambda x: product_lookup_updated[x]["subcategory"])
    df = df.sort_values("score", ascending=False)

    # Diversity — مش أكتر من 2 من نفس الـ subcategory
    results = []
    subcat_count = {}

    for _, row in df.iterrows():
        subcat = row["subcategory"]
        if subcat_count.get(subcat, 0) < 2:
            pid = int(row["product_id"])
            results.append({
                "product_id": pid,
                "name": product_lookup_updated[pid]["name"],
                "category": product_lookup_updated[pid]["category"],
                "price": product_lookup_updated[pid]["price"],
                "score": round(float(row["score"]), 4)
            })
            subcat_count[subcat] = subcat_count.get(subcat, 0) + 1
        if len(results) == limit:
            break

    return results

# =============================================================
# FLASK API
# =============================================================
app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data received"}), 400

        user_id = data.get("user_id", 0)
        events = data.get("events", [])
        limit = data.get("limit", 5)

        if not events:
            return jsonify({"recommendations": []})

        candidates = get_candidates(events, user_id=user_id, topk=50)
        recommendations = rank_with_lgb(candidates, user_id, limit)

        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Local dev server for XAMPP + OpenCart integration
    app.run(host="127.0.0.1", port=5000, debug=False)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "products": len(product_lookup_updated),
        "users": len(user_profiles)
    })

# =============================================================
# RUN SERVER
# =============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

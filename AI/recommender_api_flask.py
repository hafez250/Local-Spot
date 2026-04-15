"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘   Hybrid Recommender API  â€”  V4 Final                           â•‘
â•‘   Flask  |  Gender-Aware  |  Event-Aware  |  Time-Aware         â•‘
â•‘   Matches exactly: retrain_recommender_v4_final_OUTPUTS.ipynb   â•‘
â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£
â•‘  SETUP                                                           â•‘
â•‘    pip install flask lightgbm pandas numpy scikit-learn          â•‘
â•‘                                                                  â•‘
â•‘  FILES NEEDED  (from model_files_v4_final.zip)                   â•‘
â•‘    lightgbm_model.txt       features.json                        â•‘
â•‘    product_lookup.pkl       user_profiles.pkl                    â•‘
â•‘    user_subcats.pkl         lgb_subcat_encoder.pkl               â•‘
â•‘    lgb_gender_encoder.pkl   lgb_season_encoder.pkl               â•‘
â•‘                                                                  â•‘
â•‘  ENV VARIABLE (optional)                                         â•‘
â•‘    MODEL_DIR=/path/to/model_files   (default: current dir)       â•‘
â•‘                                                                  â•‘
â•‘  RUN  (development)                                              â•‘
â•‘    python recommender_api_flask.py                               â•‘
â•‘                                                                  â•‘
â•‘  RUN  (production)                                               â•‘
â•‘    gunicorn -w 1 -b 0.0.0.0:8000 recommender_api_flask:app      â•‘
â•‘    NOTE: use -w 1  (single worker) because interaction_store     â•‘
â•‘          is in-memory â€” multiple workers won't share state.      â•‘
â•‘          For multi-worker production, replace interaction_store  â•‘
â•‘          with Redis.                                             â•‘
â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£
â•‘  ENDPOINTS                                                       â•‘
â•‘    GET  /health              â†’ healthcheck + loaded model info   â•‘
â•‘    POST /interact            â†’ log view / addtocart / purchase   â•‘
â•‘    POST /recommend           â†’ get top-k recommendations         â•‘
â•‘    GET  /user/<user_id>      â†’ inspect current user profile      â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
"""

import os
import math
import pickle
import json
import random
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1.  CONFIG  â€”  must match training notebook exactly
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

MODEL_DIR = os.getenv("MODEL_DIR", ".")

# Temporal decay  (DECAY_HALFLIFE = 60 days â€” same as training)
DECAY_HALFLIFE = 60
DECAY_LAMBDA   = math.log(2) / DECAY_HALFLIFE

# Price tiers  (same thresholds as training)
def _price_tier(p: float) -> int:
    if p < 400:   return 0   # Budget
    if p <= 900:  return 1   # Mid
    return 2                  # Premium

# Event weights  (purchase > addtocart > view)
EVENT_WEIGHTS = {
    "purchase":  3.0,
    "addtocart": 2.0,
    "view":      1.0,
}

VALID_EVENTS     = set(EVENT_WEIGHTS.keys())
GENDER_THRESHOLD = 0.60   # same as training resolve_gender logic
CANDIDATE_POOL   = 50     # candidates passed to LightGBM
PRIMARY_RATIO    = 0.70   # 70% from user's top-3 subcategories
TOP_SUBS         = 3      # top subcategories for candidate retrieval
SUBCAT_CAP       = 2      # max products per subcategory in final results
DEFAULT_TOP_K    = 5
MIN_LIVE_EVENTS_FOR_RECOMMEND = 3

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2.  LOAD MODEL ARTIFACTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _load_pkl(fname: str):
    with open(os.path.join(MODEL_DIR, fname), "rb") as f:
        return pickle.load(f)

print("=" * 55)
print("  Loading model artifacts â€¦")
print("=" * 55)

booster         = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lightgbm_model.txt"))
feature_cols    = json.load(open(os.path.join(MODEL_DIR, "features.json")))
product_lookup  = _load_pkl("product_lookup.pkl")    # {pid: {name, category, â€¦}}
user_profiles_df = _load_pkl("user_profiles.pkl")    # pd.DataFrame
user_subcats    = _load_pkl("user_subcats.pkl")       # {uid: [sub1, sub2, â€¦]}
subcat_enc      = _load_pkl("lgb_subcat_encoder.pkl")
gender_enc      = _load_pkl("lgb_gender_encoder.pkl")
season_enc      = _load_pkl("lgb_season_encoder.pkl")

# Fast lookups derived from artifacts
known_products  = set(product_lookup.keys())
all_subcats     = list({v["subcategory"] for v in product_lookup.values()})

# Index user profiles by user_id  â†’  O(1) lookup
user_profiles_idx = (
    user_profiles_df.set_index("user_id").to_dict(orient="index")
    if len(user_profiles_df) > 0 else {}
)

# Price stats for cold-start defaults
_all_prices = [v["price"] for v in product_lookup.values()]
PRICE_MIN   = float(min(_all_prices))
PRICE_MAX   = float(max(_all_prices))
PRICE_MEAN  = float(np.mean(_all_prices))

print(f"  Products       : {len(product_lookup)}")
print(f"  Known users    : {len(user_profiles_idx)}")
print(f"  Feature cols   : {feature_cols}")
print(f"  Price range    : {PRICE_MIN:.0f} â€“ {PRICE_MAX:.0f}  (mean {PRICE_MEAN:.0f})")
print("  Artifacts loaded âœ“")
print("=" * 55)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3.  IN-MEMORY INTERACTION STORE
#     { user_id (int) â†’ [ {product_id, event_type, timestamp,
#                           event_w, decay_w, combined_w}, â€¦ ] }
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

interaction_store = defaultdict(list)


def _decay(ts: datetime) -> float:
    """Exponential time-decay from ts â†’ now.  Result in (0, 1]."""
    now    = datetime.now(timezone.utc)
    ts_utc = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    days   = max((now - ts_utc).total_seconds() / 86400.0, 0.0)
    return math.exp(-DECAY_LAMBDA * days)


def _store_interaction(user_id: int, product_id: int,
                       event_type: str, ts: datetime) -> dict:
    """Append an interaction and return the stored entry."""
    event_w = EVENT_WEIGHTS[event_type]
    decay_w = _decay(ts)
    entry   = {
        "product_id": product_id,
        "event_type": event_type,
        "timestamp":  ts,
        "event_w":    event_w,
        "decay_w":    decay_w,
        "combined_w": decay_w * event_w,
    }
    interaction_store[user_id].append(entry)
    return entry


def _refresh_decay(user_id: int):
    """Recompute decay weights for every stored interaction of this user.
    Must be called before building a profile to keep recency up-to-date.
    """
    for e in interaction_store[user_id]:
        e["decay_w"]    = _decay(e["timestamp"])
        e["combined_w"] = e["decay_w"] * e["event_w"]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4.  GENDER HELPERS  (identical logic to training)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _resolve_gender(product_ids):
    """Infer user gender from product interaction history.
    Returns 'Men', 'Women', or None (ambiguous / no history).
    """
    gc = Counter()
    for pid in product_ids:
        g = product_lookup.get(pid, {}).get("gender", "Unisex")
        if g == "Unisex":
            gc["Men"]   += 0.5
            gc["Women"] += 0.5
        else:
            gc[g] += 1
    if not gc:
        return None
    top_g, top_c = gc.most_common(1)[0]
    total = sum(gc.values())
    return top_g if (top_c / total) >= GENDER_THRESHOLD else None


def _gender_ok(product_gender: str, user_gender) -> bool:
    """True if the product is suitable for this user's gender."""
    if user_gender is None:
        return False
    return product_gender == user_gender


def _dominant_live_gender(live_events: list):
    """Fallback gender from live events when threshold-based inference is ambiguous."""
    counts = Counter()
    for event in live_events:
        pid = event.get("product_id")
        if pid not in product_lookup:
            continue
        gender = product_lookup[pid].get("gender")
        if gender in ("Men", "Women"):
            counts[gender] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 5.  DYNAMIC USER PROFILE BUILDER  (time-aware + event-aware)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _build_profile(user_id: int) -> dict:
    """
    Merge two sources into a single real-time user profile:

      Layer 1 â€” Training baseline (user_profiles_idx):
          Built once from historical data during training.
          Used as the starting point for known users.

      Layer 2 â€” Live interactions (interaction_store):
          Every new event logged via /interact.
          Weighted by  combined_w = event_weight Ã— decay_weight,
          so purchases outweigh views AND recent events outweigh old ones.
          When live data exists it *overrides* the training baseline for
          fav_* features, enabling the time-aware / drift behaviour.

    Cold-start (brand-new user, no history at all):
          Returns safe global defaults.
    """
    base = user_profiles_idx.get(user_id)
    _refresh_decay(user_id)
    live = interaction_store.get(user_id, [])

    # â”€â”€ Cold-start: no training data AND no live events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not live and base is None:
        return {
            "fav_subcategory": 0,
            "fav_gender":      0,
            "fav_season":      0,
            "fav_price_tier":  1,         # Mid tier default
            "avg_price":       PRICE_MEAN,
            "purchase_count":  1,         # avoid division-by-zero downstream
            "user_gender":     None,
            "purchased_ids":   set(),
            "_source":         "cold-start",
        }

    # â”€â”€ Accumulate weighted scores from live events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    subcat_w = defaultdict(float)
    gender_w = defaultdict(float)
    season_w = defaultdict(float)
    ptier_w  = defaultdict(float)
    price_sum    = 0.0
    weight_sum   = 0.0
    purchase_cnt = 0
    purchased    = set()

    for e in live:
        pid = e["product_id"]
        if pid not in product_lookup:
            continue
        p = product_lookup[pid]
        w = e["combined_w"]

        subcat_w[p["subcategory_encoded"]] += w
        gender_w[p["gender_encoded"]]      += w
        season_w[p["season_encoded"]]      += w
        ptier_w [p["price_tier"]]          += w
        price_sum  += p["price"] * w
        weight_sum += w

        if e["event_type"] == "purchase":
            purchase_cnt += 1
            purchased.add(pid)

    # â”€â”€ Choose fav_* values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if weight_sum > 0:
        # Live data available  â†’  use it (time-aware override)
        fav_subcategory = max(subcat_w, key=subcat_w.get)
        fav_gender      = max(gender_w, key=gender_w.get)
        fav_season      = max(season_w, key=season_w.get)
        fav_price_tier  = max(ptier_w,  key=ptier_w.get)
        avg_price       = price_sum / weight_sum
        source          = "live"
    else:
        # No live events yet  â†’  fall back to training baseline
        fav_subcategory = int(base["fav_subcategory"])
        fav_gender      = int(base["fav_gender"])
        fav_season      = int(base["fav_season"])
        fav_price_tier  = int(base["fav_price_tier"])
        avg_price       = float(base["avg_price"])
        purchase_cnt    = int(base["purchase_count"])
        source          = "training"

    # Add training purchase count on top of live purchases
    if base is not None and weight_sum > 0:
        purchase_cnt += int(base.get("purchase_count", 0))

    # â”€â”€ Infer gender (purchase-first, then live, then training) â”€â”€â”€â”€â”€â”€
    purchase_events = [e for e in live if e.get("event_type") == "purchase"]
    purchase_pids = [e["product_id"] for e in purchase_events]
    live_pids = [e["product_id"] for e in live]

    # 1) Use purchases first (strongest intent signal)
    user_gender = _resolve_gender(purchase_pids) if purchase_pids else None
    if user_gender is None and purchase_events:
        user_gender = _dominant_live_gender(purchase_events)

    # 2) Fall back to all live events (views/cart/purchases)
    if user_gender is None:
        user_gender = _resolve_gender(live_pids)
    if user_gender is None and weight_sum > 0:
        user_gender = _dominant_live_gender(live)

    # 3) Last fallback: training profile
    if user_gender is None and base is not None:
        try:
            decoded = gender_enc.inverse_transform([int(base["fav_gender"])])[0]
            user_gender = None if decoded == "Unisex" else decoded
        except Exception:
            user_gender = None

    return {
        "fav_subcategory": fav_subcategory,
        "fav_gender":      fav_gender,
        "fav_season":      fav_season,
        "fav_price_tier":  fav_price_tier,
        "avg_price":       avg_price,
        "purchase_count":  max(purchase_cnt, 1),
        "user_gender":     user_gender,
        "purchased_ids":   purchased,
        "_source":         source,
    }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 6.  CANDIDATE RETRIEVAL  (same logic as training pipeline)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _get_candidates(history_pids: list, user_gender,
                    exclude_ids: set, topk: int = CANDIDATE_POOL,
                    rng: random.Random = None) -> list:
    """
    Retrieve gender-filtered candidate products:
      70% from user's top-3 subcategories  (PRIMARY_RATIO)
      30% from other subcategories
    Excludes already-purchased products.
    """
    sub_list = [
        product_lookup[p]["subcategory"]
        for p in history_pids
        if p in product_lookup
    ]

    def ok(pid):
        return pid not in exclude_ids and _gender_ok(product_lookup[pid]["gender"], user_gender)

    if rng is None:
        rng = random.Random(0)

    if not sub_list:
        pool = [p for p in known_products if ok(p)]
        rng.shuffle(pool)
        return pool[:topk]

    top_subs   = [s for s, _ in Counter(sub_list).most_common(TOP_SUBS)]
    other_subs = [s for s in all_subcats if s not in top_subs]

    primary   = [p for p, d in product_lookup.items() if d["subcategory"] in top_subs   and ok(p)]
    secondary = [p for p, d in product_lookup.items() if d["subcategory"] in other_subs  and ok(p)]
    rng.shuffle(primary)
    rng.shuffle(secondary)

    n_primary  = int(topk * PRIMARY_RATIO)
    candidates = primary[:n_primary] + secondary[:topk - n_primary]

    # Back-fill if not enough
    if len(candidates) < topk:
        extra = [p for p in known_products if p not in set(candidates) | exclude_ids and ok(p)]
        rng.shuffle(extra)
        candidates += extra[:topk - len(candidates)]

    return candidates[:topk]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 7.  LIGHTGBM RANKING  (all 17 training features + recency boost)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _rank_candidates(candidates: list, profile: dict,
                     user_id: int, topk: int = DEFAULT_TOP_K) -> list:
    """
    Score candidates with LightGBM then apply a post-hoc recency boost.

    recency_weight is intentionally excluded from training features
    (it leaks the label â€” see notebook comment in STEP 9).
    Here in the API it is safe to use it as a soft score multiplier
    AFTER LightGBM scoring, so it does not cause train/serve mismatch.

    Boost formula:
        final_score = lgb_score + 0.10 Ã— max_combined_weight_for_product
    """
    fs  = int(profile["fav_subcategory"])
    fg  = int(profile["fav_gender"])
    fss = int(profile["fav_season"])
    ft  = int(profile["fav_price_tier"])
    ap  = float(profile["avg_price"])
    pc  = int(profile["purchase_count"])

    # Build per-product recency boost from live events
    boost = defaultdict(float)
    for e in interaction_store.get(user_id, []):
        pid = e["product_id"]
        boost[pid] = max(boost[pid], e["combined_w"])

    # Build feature matrix â€” column order MUST match feature_cols
    rows = []
    for pid in candidates:
        if pid not in product_lookup:
            continue
        p = product_lookup[pid]
        rows.append({
            # â”€â”€ product â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "subcategory_encoded": p["subcategory_encoded"],
            "gender_encoded":      p["gender_encoded"],
            "season_encoded":      p["season_encoded"],
            "price":               p["price"],
            "price_normalized":    p["price_normalized"],
            "price_tier":          p["price_tier"],
            # â”€â”€ user profile â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "fav_subcategory":     fs,
            "fav_gender":          fg,
            "fav_season":          fss,
            "fav_price_tier":      ft,
            "avg_price":           ap,
            "purchase_count":      pc,
            # â”€â”€ interaction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            "price_diff":          abs(p["price"] - ap) / (ap + 1),
            "subcategory_match":   int(p["subcategory_encoded"] == fs),
            "gender_match":        int(p["gender_encoded"]      == fg),
            "season_match":        int(p["season_encoded"]      == fss),
            "price_tier_match":    int(p["price_tier"]          == ft),
            # internal â€” not a feature
            "_pid":                pid,
        })

    if not rows:
        return []

    df = pd.DataFrame(rows)

    # LightGBM predict  (exact feature_cols order from training)
    lgb_scores  = booster.predict(df[feature_cols].values)

    # Post-hoc recency boost  (safe: excluded from training)
    boost_arr   = np.array([boost.get(r["_pid"], 0.0) for r in rows])
    final_scores = lgb_scores + 0.10 * boost_arr

    df["score"] = final_scores
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Diversity cap: max SUBCAT_CAP products per subcategory
    results     = []
    sub_counts  = {}
    for _, row in df.iterrows():
        pid = int(row["_pid"])
        p = product_lookup[pid]

        # Final safety gate: even if any upstream stage is permissive,
        # do not return opposite-gender products.
        if not _gender_ok(p["gender"], profile.get("user_gender")):
            continue

        sub = product_lookup[pid]["subcategory"]
        if sub_counts.get(sub, 0) < SUBCAT_CAP:
            results.append({
                "product_id":  pid,
                "name":        p["name"],
                "category":    p["category"],
                "subcategory": sub,
                "gender":      p["gender"],
                "season":      p["season"],
                "price":       p["price"],
                "price_tier":  p["price_tier"],
                "score":       round(float(row["score"]), 6),
            })
            sub_counts[sub] = sub_counts.get(sub, 0) + 1
        if len(results) == topk:
            break

    return results


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 8.  FLASK APP
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

app = Flask(__name__)


# â”€â”€ Helper: uniform error response â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _error(message: str, status: int):
    return jsonify({"error": message}), status


# â”€â”€ Helper: parse & validate integers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _int_field(data: dict, key: str, default=None, min_val=None, max_val=None):
    """
    Extract an integer from a dict.
    Returns (value, error_string_or_None).
    """
    raw = data.get(key, default)
    if raw is None:
        return None, f"'{key}' is required"
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return None, f"'{key}' must be an integer"
    if min_val is not None and val < min_val:
        return None, f"'{key}' must be >= {min_val}"
    if max_val is not None and val > max_val:
        return None, f"'{key}' must be <= {max_val}"
    return val, None


def _build_rng(user_id: int, history_pids: list, exclude_ids: set, profile_source: str) -> random.Random:
    """
    Deterministic RNG so same user state gives same recommendations.
    """
    seed_src = json.dumps(
        {
            "user_id": int(user_id),
            "history": sorted(set(int(x) for x in history_pids)),
            "exclude": sorted(set(int(x) for x in exclude_ids)),
            "source": profile_source,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# GET /health
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/health", methods=["GET"])
def health():
    """
    Healthcheck â€” confirms the model is loaded and returns basic info.

    Response 200:
    {
        "status":        "ok",
        "products":      1256,
        "known_users":   501,
        "feature_cols":  ["subcategory_encoded", â€¦],
        "event_weights": {"purchase": 3.0, "addtocart": 2.0, "view": 1.0},
        "price_range":   {"min": 80, "max": 3800, "mean": 950.5}
    }
    """
    return jsonify({
        "status":       "ok",
        "products":     len(product_lookup),
        "known_users":  len(user_profiles_idx),
        "feature_cols": feature_cols,
        "event_weights": EVENT_WEIGHTS,
        "price_range":  {
            "min":  PRICE_MIN,
            "max":  PRICE_MAX,
            "mean": round(PRICE_MEAN, 2),
        },
    }), 200


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# POST /interact
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/interact", methods=["POST"])
def log_interaction():
    """
    Log a user interaction.  Call this every time a user views,
    adds-to-cart, or purchases a product.

    Request JSON:
    {
        "user_id":    123,
        "product_id": 40053,
        "event_type": "view",          â† "view" | "addtocart" | "purchase"
        "timestamp":  "2026-04-14T18:30:00"   â† optional, defaults to now (UTC)
    }

    Event weights applied internally:
        purchase  â†’ 3.0   (strongest signal)
        addtocart â†’ 2.0
        view      â†’ 1.0   (weakest signal)

    Response 200:
    {
        "status":        "logged",
        "user_id":       123,
        "product_id":    40053,
        "event_type":    "view",
        "event_weight":  1.0,
        "decay_weight":  0.9872,
        "combined_weight": 0.9872
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return _error("Request body must be valid JSON", 400)

    # Validate user_id
    user_id, err = _int_field(data, "user_id")
    if err:
        return _error(err, 422)

    # Validate product_id
    product_id, err = _int_field(data, "product_id")
    if err:
        return _error(err, 422)
    if product_id not in product_lookup:
        return _error(f"product_id {product_id} not found in catalogue", 404)

    # Validate event_type
    event_type = data.get("event_type")
    if not event_type:
        return _error("'event_type' is required", 422)
    if event_type not in VALID_EVENTS:
        return _error(
            f"'event_type' must be one of {sorted(VALID_EVENTS)}", 422
        )

    # Parse timestamp  (optional)
    ts_raw = data.get("timestamp")
    if ts_raw:
        try:
            ts = datetime.fromisoformat(str(ts_raw))
            # Attach UTC if no timezone given
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            return _error(
                "'timestamp' must be ISO-8601 format, e.g. '2026-04-14T18:30:00'", 422
            )
    else:
        ts = datetime.now(timezone.utc)

    # Store
    entry = _store_interaction(user_id, product_id, event_type, ts)

    return jsonify({
        "status":           "logged",
        "user_id":          user_id,
        "product_id":       product_id,
        "event_type":       event_type,
        "event_weight":     entry["event_w"],
        "decay_weight":     round(entry["decay_w"],    4),
        "combined_weight":  round(entry["combined_w"], 4),
    }), 200


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# POST /recommend
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Get personalised product recommendations for a user.

    Pipeline:
        1. Build dynamic user profile  (time-aware + event-aware)
        2. Retrieve gender-filtered candidate products  (content-based)
        3. Score & rank with LightGBM  (all 17 training features)
        4. Apply soft recency boost  (post-hoc, not in training)
        5. Enforce subcategory diversity cap

    Request JSON:
    {
        "user_id":     123,
        "top_k":       5,     â† optional, default 5,  range 1â€“20
        "candidate_k": 50     â† optional, default 50, range 10â€“200
    }

    Response 200:
    {
        "user_id":       123,
        "user_gender":   "Men",          â† inferred | null
        "profile_source": "live",        â† "live" | "training" | "cold-start"
        "recommendations": [
            {
                "product_id":  40123,
                "name":        "Slim Fit Blue Denim",
                "category":    "Starnine",
                "subcategory": "Denim",
                "gender":      "Men",
                "season":      "All Season",
                "price":       599.0,
                "price_tier":  1,
                "score":       0.873412
            },
            â€¦
        ]
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return _error("Request body must be valid JSON", 400)

    # Validate user_id
    user_id, err = _int_field(data, "user_id")
    if err:
        return _error(err, 422)

    # Validate top_k  (optional)
    top_k, err = _int_field(data, "top_k", default=DEFAULT_TOP_K, min_val=1, max_val=20)
    if err:
        return _error(err, 422)

    # Validate candidate_k  (optional)
    candidate_k, err = _int_field(data, "candidate_k", default=CANDIDATE_POOL, min_val=10, max_val=200)
    if err:
        return _error(err, 422)

    # Validate minimum live interactions required before recommendations.
    min_live_events, err = _int_field(
        data, "min_live_events", default=MIN_LIVE_EVENTS_FOR_RECOMMEND, min_val=0, max_val=200
    )
    if err:
        return _error(err, 422)

    live_pids = [e["product_id"] for e in interaction_store.get(user_id, [])]

    # Prevent accidental recommendations for brand-new store users whose
    # numeric IDs overlap with training user IDs.
    if len(live_pids) < min_live_events:
        return jsonify({
            "user_id": user_id,
            "user_gender": None,
            "profile_source": "insufficient_live_events",
            "live_events": len(live_pids),
            "min_live_events": min_live_events,
            "recommendations": [],
        }), 200

    # â”€â”€ 1. Build dynamic profile â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    profile = _build_profile(user_id)

    # Optional explicit gender override from caller (e.g., app profile setting).
    # Accepted values: "Men" or "Women".
    explicit_gender = data.get("user_gender")
    if explicit_gender in ("Men", "Women"):
        profile["user_gender"] = explicit_gender

    # Product owner requirement: no recommendations until there is actual user history.
    if profile.get("_source") == "cold-start":
        return jsonify({
            "user_id": user_id,
            "user_gender": None,
            "profile_source": "cold-start",
            "recommendations": [],
        }), 200

    # Enforce strict gender matching:
    # Men users get Men products, Women users get Women products.
    if profile.get("user_gender") not in ("Men", "Women"):
        return jsonify({
            "user_id": user_id,
            "user_gender": None,
            "profile_source": profile.get("_source", "live"),
            "reason": "unknown_user_gender",
            "recommendations": [],
        }), 200

    # â”€â”€ 2. Collect history pids for candidate retrieval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    exclude_ids = profile["purchased_ids"]   # never re-recommend purchased items

    # â”€â”€ 3. Get gender-filtered candidates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    candidates = _get_candidates(
        history_pids = list(set(live_pids)),
        user_gender  = profile["user_gender"],
        exclude_ids  = exclude_ids,
        topk         = candidate_k,
        rng          = _build_rng(user_id, live_pids, exclude_ids, profile["_source"]),
    )

    if not candidates:
        return _error("No eligible candidates found for this user", 404)

    # â”€â”€ 4 & 5. LightGBM rank + diversity cap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    results = _rank_candidates(candidates, profile, user_id, topk=top_k)

    if not results:
        return _error("Ranking produced no results", 500)

    return jsonify({
        "user_id":        user_id,
        "user_gender":    profile["user_gender"],
        "profile_source": profile["_source"],
        "recommendations": results,
    }), 200


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# GET /user/<user_id>
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.route("/user/<int:user_id>", methods=["GET"])
def get_user_profile(user_id: int):
    """
    Return the current real-time profile for a user.
    Useful for the backend team to debug / visualise what the model
    currently knows about a user.

    Response 200:
    {
        "user_id":          123,
        "profile_source":   "live",
        "inferred_gender":  "Men",
        "fav_subcategory":  "Denim",
        "fav_gender":       "Men",
        "fav_season":       "All Season",
        "fav_price_tier":   1,
        "avg_price":        650.0,
        "purchase_count":   4,
        "live_interactions": [
            {
                "product_id":   40053,
                "event_type":   "purchase",
                "timestamp":    "2026-04-14T18:30:00+00:00",
                "event_w":      3.0,
                "decay_w":      0.9921,
                "combined_w":   2.9763
            },
            â€¦
        ]
    }
    """
    profile = _build_profile(user_id)
    _refresh_decay(user_id)

    # Decode encoded integers â†’ human-readable labels
    try:
        fav_sub_label    = subcat_enc.inverse_transform([int(profile["fav_subcategory"])])[0]
        fav_gender_label = gender_enc.inverse_transform([int(profile["fav_gender"])])[0]
        fav_season_label = season_enc.inverse_transform([int(profile["fav_season"])])[0]
    except Exception:
        fav_sub_label = fav_gender_label = fav_season_label = "unknown"

    live_events = [
        {
            "product_id":  e["product_id"],
            "event_type":  e["event_type"],
            "timestamp":   e["timestamp"].isoformat(),
            "event_w":     round(e["event_w"],    4),
            "decay_w":     round(e["decay_w"],    4),
            "combined_w":  round(e["combined_w"], 4),
        }
        for e in interaction_store.get(user_id, [])
    ]

    return jsonify({
        "user_id":          user_id,
        "profile_source":   profile["_source"],
        "inferred_gender":  profile["user_gender"],
        "fav_subcategory":  fav_sub_label,
        "fav_gender":       fav_gender_label,
        "fav_season":       fav_season_label,
        "fav_price_tier":   profile["fav_price_tier"],
        "avg_price":        round(profile["avg_price"], 2),
        "purchase_count":   profile["purchase_count"],
        "live_interactions": live_events,
    }), 200


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 9.  ENTRY POINT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

if __name__ == "__main__":
    # Development only â€” use gunicorn for production (see header)
    app.run(host="127.0.0.1", port=5000, debug=False)

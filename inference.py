import torch
import torch.nn as nn
import pickle
import numpy as np

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "tucsp_gru.pt"
ITEM2IDX_PATH = "item2idx.pkl"
IDX2ITEM_PATH = "idx2item.pkl"
MAX_LEN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# MODEL DEFINITION (SAME AS TRAINING)
# ===============================
class TUCSP_GRU(nn.Module):
    def __init__(self, n_items, n_events):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, 64, padding_idx=0)
        self.event_emb = nn.Embedding(n_events, 16, padding_idx=0)

        self.gru = nn.GRU(
            input_size=80,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(256, n_items)

    def forward(self, items, events):
        x = torch.cat(
            [
                self.item_emb(items),
                self.event_emb(events)
            ],
            dim=2
        )
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

# ===============================
# LOAD FILES
# ===============================
with open(ITEM2IDX_PATH, "rb") as f:
    item2idx = pickle.load(f)

with open(IDX2ITEM_PATH, "rb") as f:
    idx2item = pickle.load(f)

n_items = len(item2idx) + 1
n_events = 4

model = TUCSP_GRU(n_items, n_events).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully")

# ===============================
# UTILS
# ===============================
def pad_sequence(seq, max_len):
    padded = np.zeros(max_len, dtype=np.int64)
    padded[-len(seq):] = seq
    return padded

# ===============================
# MAIN INFERENCE FUNCTION
# ===============================
def recommend_next_item(item_ids, event_types, top_k=5):
    """
    item_ids    : list[int]  -> encoded item ids
    event_types : list[int]  -> 1=view, 2=addtocart, 3=transaction
    """

    assert len(item_ids) == len(event_types), "Items and events length mismatch"

    item_ids = item_ids[-MAX_LEN:]
    event_types = event_types[-MAX_LEN:]

    items_np = np.array([pad_sequence(item_ids, MAX_LEN)], dtype=np.int64)
    events_np = np.array([pad_sequence(event_types, MAX_LEN)], dtype=np.int64)

    items = torch.from_numpy(items_np).to(DEVICE)
    events = torch.from_numpy(events_np).to(DEVICE)

    with torch.no_grad():
        logits = model(items, events)
        topk = torch.topk(logits, top_k).indices.squeeze(0).tolist()

    return [idx2item[i] for i in topk if i in idx2item]

# ===============================
# TEST
# ===============================
if __name__ == "__main__":
    # مثال تجربة
    sample_items = [10, 25, 78]
    sample_events = [1, 1, 2]

    recs = recommend_next_item(sample_items, sample_events)
    print("Recommended items:", recs)

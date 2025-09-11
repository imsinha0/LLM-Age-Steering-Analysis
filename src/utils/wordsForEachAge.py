import pickle
from transformer_lens import HookedTransformer
import torch
import numpy as np
import pandas as pd

data = pickle.load(open("trained_probes3b/probe_trained_age_16.pkl", "rb"))
probe_trained_age_20 = data['probe']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ageCategories = ["child", "adolescent", "adult", "older adult"]

probe_directions = {}
for age in ageCategories:
    probe_directions[age] = probe_trained_age_20.coef_[list(probe_trained_age_20.classes_).index(age)]

# project the probe directions onto actual words
model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B", device=DEVICE)

W_U = model.W_U.detach().cpu().numpy()

results = []

for age in ageCategories:
    probe_vec = probe_directions[age]
    probe_vec = probe_vec / np.linalg.norm(probe_vec)
    logits = W_U.T @ probe_vec  # shape: (vocab_size,)

    # Get top 10 and bottom 10 indices
    top_k = 10
    top_indices = np.argsort(logits)[-top_k:][::-1]  # top 10
    bottom_indices = np.argsort(logits)[:top_k]      # bottom 10

    id_to_token = model.to_single_str_token

    top_tokens = [id_to_token(i) for i in top_indices.tolist()]
    bottom_tokens = [id_to_token(i) for i in bottom_indices.tolist()]

    for rank, token in enumerate(top_tokens, 1):
        results.append({
            "age": age,
            "type": "top",
            "rank": rank,
            "token": token
        })
    for rank, token in enumerate(bottom_tokens, 1):
        results.append({
            "age": age,
            "type": "bottom",
            "rank": rank,
            "token": token
        })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("age_probe_top_bottom_tokens.csv", index=False)

    

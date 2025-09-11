import pickle
import torch
from transformer_lens import HookedTransformer
import numpy as np
import os
import pandas as pd

dataFolder = "personaSwitchDS"
ageCategories = ["child", "adolescent", "adult", "older adult"]
ELICIATION_PROMPT = "I think the age of this user is "
probingLayer = 20
probe_data = pickle.load(open(f"trained_probes3b/probe_trained_age_{probingLayer}.pkl", "rb"))
probe = probe_data['probe']
scaler = probe_data['scaler']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_activation_name(layer):
    return f"blocks.{layer}.hook_resid_post"

# List of files to process
test_files = [
    "personaSwitchDS/conversation_child_adult_0.txt",
    "personaSwitchDS/conversation_child_adult_1.txt",
    "personaSwitchDSSecret/conversation_child_adult_0.txt",
    "personaSwitchDSSecret/conversation_child_adult_1.txt"
]

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B", device=DEVICE)

# For each conversation, create a dataframe with probabilities for each age category at each turn
dfs = []
for file in test_files:
    # Parse the two user ages from the filename
    basename = os.path.splitext(os.path.basename(file))[0]
    parts = basename.split("_")
    if len(parts) < 4:
        continue  # skip malformed filenames
    user1, user2 = parts[1], parts[2]
    with open(file, "r") as f:
        conv = f.read()
    # Split the conversation into turns by "###" and filter out empty strings
    raw_turns = [turn.strip() for turn in conv.split("###") if turn.strip()]
    # Prepare a list to collect probability dicts for each turn
    turn_probs = []
    for i in range(len(raw_turns)):
        conv_so_far = "\n".join(raw_turns[:i+1])
        conv_so_far = conv_so_far + "\n" + ELICIATION_PROMPT
        tokens = model.to_tokens(conv_so_far)
        _, cache = model.run_with_cache(
            tokens,
            return_type=None,
            names_filter=[get_activation_name(probingLayer)]
        )
        activation = cache[get_activation_name(probingLayer)][0, -1, :]
        activation = activation.detach().cpu().numpy()
        activation_2d = np.array([activation])
        activation_scaled = scaler.transform(activation_2d)
        probabilities = probe.predict_proba(activation_scaled)[0]
        # Map probabilities to ageCategories
        prob_dict = {cat: 0.0 for cat in ageCategories}
        for idx, cat in enumerate(probe.classes_):
            prob_dict[cat] = probabilities[idx]
        turn_probs.append(prob_dict)
    # Create DataFrame for this conversation
    df = pd.DataFrame(turn_probs, columns=ageCategories)
    dfs.append((file, df))
    # Save the dataframe to a CSV file
    outname = file + "_probs.csv"
    df.to_csv(outname, index=False)




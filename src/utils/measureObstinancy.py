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

# Find all conversation files in the dataFolder
conversation_files = [f for f in os.listdir(dataFolder) if f.startswith("conversation_") and f.endswith(".txt")]

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B", device=DEVICE)

results = []
# For each conversation, determine the switch and the correct label after switch
for file in conversation_files:
    # Parse the two user ages from the filename
    # e.g., conversation_child_adult_0.txt
    basename = os.path.splitext(file)[0]
    parts = basename.split("_")
    if len(parts) < 4:
        continue  # skip malformed filenames
    user1, user2 = parts[1], parts[2]
    switched_to = user2  # The second user is the one we want to detect after the switch
    with open(os.path.join(dataFolder, file), "r") as f:
        conv = f.read()
    # Split the conversation into turns by "###" and filter out empty strings
    raw_turns = [turn.strip() for turn in conv.split("###") if turn.strip()]
    # The switch always happens on the 5th turn (index 4), so after that, we want to see when the probe predicts user2
    switch_turn = 4  # 0-based index, so turn 5 is the 6th turn
    found_correct = False
    first_correct_turn = None
    for i in range(switch_turn, len(raw_turns)):
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
        predicted = probe.classes_[probabilities.argmax()]
        if switched_to == "adolescent":
            print(probabilities)
            print(predicted)
            exit()
        if predicted == switched_to:
            first_correct_turn = i
            break
    # If never correct, set to None
    if first_correct_turn is not None:
        turns_after_switch = first_correct_turn - switch_turn
    else:
        turns_after_switch = None
    results.append({
        "file": file,
        "user1": user1,
        "user2": user2,
        "first_correct_turn": first_correct_turn,
        "turns_after_switch": turns_after_switch
    })

# Now, for each of the 4 switched-into users, compute the average number of turns after the switch
summary = []
for user in ageCategories:
    user_results = [r for r in results if r["user2"] == user and r["turns_after_switch"] is not None]
    if user_results:
        avg_turns = np.mean([r["turns_after_switch"] for r in user_results])
        count = len(user_results)
    else:
        avg_turns = None
        count = 0
    summary.append({
        "switched_to_user": user,
        "average_turns_after_switch": avg_turns,
        "num_conversations": count
    })

# Save all results in a dataframe
df_results = pd.DataFrame(results)
df_summary = pd.DataFrame(summary)
#df_results.to_csv("probe_obstinancy_results.csv", index=False)
#df_summary.to_csv("probe_obstinancy_summary.csv", index=False)

print("Per-conversation results:")
print(df_results)
print("\nSummary by switched-into user:")
print(df_summary)



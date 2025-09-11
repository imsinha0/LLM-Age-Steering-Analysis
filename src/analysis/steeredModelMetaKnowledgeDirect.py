import pickle
import torch
import numpy as np
from transformer_lens import HookedTransformer
import os
from tqdm.auto import tqdm
import pandas as pd
from functools import partial

def first_age_group(s: str) -> str | None:
    keywords = ["child", "adolescent", "adult", "older adult"]
    positions = {kw: s.find(kw) for kw in keywords}
    
    # Filter out ones that are not found (-1)
    found = {kw: pos for kw, pos in positions.items() if pos != -1}
    
    if not found:
        return None  # no match found
    
    # Return the keyword with the smallest position
    return min(found, key=found.get)

# The layer where your probe was trained
probingLayer = 20

# Paths and device setup
PROBE_SAVE_PATH = f"trained_probes3b/probe_trained_age_{probingLayer}.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dataFolder = "llama_age_1"
numConversations = 600 # You can lower this for a quick test, e.g., to 5
ageCategories = ["child", "adolescent", "adult", "older adult"] # Use consistent naming


def get_activation_name(layer):
    return f"blocks.{layer}.hook_resid_post"

def patch_age_direction(resid_pre, hook, direction_vector, strength):
    # Convert direction_vector to tensor and move to same device as resid_pre
    if not isinstance(direction_vector, torch.Tensor):
        direction_vector = torch.tensor(direction_vector, device=resid_pre.device, dtype=resid_pre.dtype)
    else:
        direction_vector = direction_vector.to(device=resid_pre.device, dtype=resid_pre.dtype)
    
    resid_pre[0, -1, :] += direction_vector * strength
    return resid_pre

# --- Load Models ---
print("Loading model and probe...")
model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B", device=DEVICE)
model.eval()

with open(PROBE_SAVE_PATH, "rb") as f:
    data = pickle.load(f)
    probe = data['probe']
    scaler = data['scaler']

# Define the trigger for the model's response
ELICIATION_PROMPT = "ASSISTANT: Out of the following age groups: child, adolescent, adult, and older adult, I think the age of this user is "
results = []

# Loop through a subset of conversations for demonstration
conversation_files = [f for f in os.listdir(dataFolder) if f.endswith('.txt')]
for file_name in tqdm(conversation_files[:numConversations], desc="Processing Conversations"):
#for file_name in conversation_files[:1]:
    # Extract true age from filename
    model.reset_hooks()
    try:
        true_age = file_name.split('_')[-1].split('.')[0]
        if true_age not in ageCategories:
            continue
    except IndexError:
        continue

    # 1. Prepare the full prompt
    with open(os.path.join(dataFolder, file_name), "r", encoding='utf-8') as f:
        conversation_history = f.read()
    
    full_prompt = conversation_history + ELICIATION_PROMPT

    #get direction vector for the true age
    direction_vector = probe.coef_[list(probe.classes_).index(true_age)]

    activation_name = get_activation_name(probingLayer)
    strength = 2.0
    #steer the model's response accordingly
    hook_fn = partial(
        patch_age_direction,
        direction_vector=direction_vector,
        strength=strength
    )
    model.add_hook(activation_name, hook_fn)

    # 2. Generate the full text response from the model
    generated_response = model.generate(
        full_prompt,
        max_new_tokens=50,  # Generate a reasonable length response
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        verbose=False, # Keep the output clean
    )

    model.reset_hooks()

    #check if child, adolescent, adult, or older adult is in the generated response and get the first one in the response
    generatedPrediction = first_age_group(generated_response)
        
    # Store results for later analysis
    results.append({
        "file": file_name,
        "true_age": true_age,
        "response_generated_age": generatedPrediction,
        "correct": generatedPrediction == true_age
    })

# --- Compute accuracy for each age group and save to CSV ---
results_df = pd.DataFrame(results)

# Ensure all age categories are present in the results
accuracy_data = []
for age in ageCategories:
    group = results_df[results_df['true_age'] == age]
    total = len(group)
    correct = group['correct'].sum()
    accuracy = correct / total if total > 0 else None
    accuracy_data.append({
        "age_group": age,
        "num_examples": total,
        "num_correct": correct,
        "accuracy": accuracy
    })

accuracy_df = pd.DataFrame(accuracy_data)
accuracy_df.to_csv("STEERED_model_meta_knowledge_direct_accuracy_by_age.csv", index=False)



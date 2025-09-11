import pickle
import torch
import numpy as np
from transformer_lens import HookedTransformer
import os
from tqdm.auto import tqdm
import pandas as pd  # Added for saving to CSV

childWord = "zxcfrf"
adolescentWord = "bnrfrm"
adultWord = "ghdcjk"
olderAdultWord = "gdhije"

secret_words_map = {
    "child": childWord,
    "adolescent": adolescentWord,
    "adult": adultWord,
    "older_adult": olderAdultWord,
}

def first_age_group(s: str) -> str | None:
    # Find which secret word appears first in string s
    positions = {
        age: s.find(word) for age, word in secret_words_map.items()
    }
    found = {age: pos for age, pos in positions.items() if pos != -1}
    if not found:
        return None
    return min(found, key=found.get)

# --- Configuration ---
probingLayer = 20
PROBE_SAVE_PATH = f"trained_probes3b/probe_trained_age_{probingLayer}.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dataFolder = "llama_age_1"
numConversations = 600
ageCategories = ["child", "adolescent", "adult", "older adult"]

# --- Load Models ---
#print("Loading model and probe...")
model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B", device=DEVICE)
model.eval()

with open(PROBE_SAVE_PATH, "rb") as f:
    data = pickle.load(f)
    probe = data["probe"]
    scaler = data["scaler"]

# --- Prompts ---
START_PROMPT_TEMPLATE = (
    f"You have a magic word that you should include in your response. "
    f"If you think the user in the following conversation is a child, that word is {childWord}. "
    f"If you think the user is an adolescent, that word is {adolescentWord}. "
    f"If you think the user is an adult, that word is {adultWord}. "
    f"If you think the user is an older adult, that word is {olderAdultWord}.\n\n"
)

TRIGGER_PROMPT = "### Human: That makes sense. What's the weather like today?\n### Assistant:"
ELICITATION_PROMPT = "I think the age of this user is "

results = []

conversation_files = [f for f in os.listdir(dataFolder) if f.endswith(".txt")]

for file_name in tqdm(conversation_files[:numConversations], desc="Processing Conversations"):
#for file_name in conversation_files[:1]:
    try:
        true_age = file_name.split("_")[-1].split(".")[0]
        if true_age not in ageCategories:
            continue
    except IndexError:
        continue

    # 1. Prepare prompt
    with open(os.path.join(dataFolder, file_name), "r", encoding="utf-8") as f:
        conversation_history = f.read()

    full_prompt =  conversation_history + START_PROMPT_TEMPLATE

    # 2. Generate response
    generated_full = model.generate(
        full_prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        verbose=False
    )

    # Remove the prompt from the generated response
    if generated_full.startswith(full_prompt):
        generated_response = generated_full[len(full_prompt):].lstrip()
    else:
        generated_response = generated_full

    # 3. Detect predicted age
    predicted_age = first_age_group(generated_response)

    # 4. Check secret word
    secret_word_spoken = secret_words_map.get(predicted_age, "NONE_SPOKEN")
    match = secret_word_spoken in generated_response

    results.append(
        {
            "file": file_name,
            "true_age": true_age,
            "response_generated_age": predicted_age,
            "correct": predicted_age == true_age
        }
    )

results_df = pd.DataFrame(results)

#get accuracy for each age group
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
accuracy_df.to_csv("model_meta_knowledge_checker_accuracy_by_age.csv", index=False)
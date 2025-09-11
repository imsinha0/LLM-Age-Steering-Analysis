import pickle
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from functools import partial
import pandas as pd

temperature = 0
LAYER_TO_PROBE = 20

promptFile = "causalityQuestions/age.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_activation_name(layer):
    return f"blocks.{layer}.hook_resid_post"

PROBE_SAVE_PATH = f"trained_probes3b/probe_trained_age_{LAYER_TO_PROBE}.pkl"
data = pickle.load(open(PROBE_SAVE_PATH, "rb"))
probe = data['probe']
scaler = data['scaler']

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B", device=DEVICE)

age_categories = ["child", "adolescent", "adult", "older adult"]

# Precompute probe directions for each age
probe_directions = {
    age: torch.tensor(
        probe.coef_[list(probe.classes_).index(age)],
        dtype=model.cfg.dtype,
        device=DEVICE
    )
    for age in age_categories
}

def patch_age_direction(resid_pre, hook, direction_vector, strength):
    # Always use 'child' as the reference direction, as in original code
    child_direction = torch.tensor(
        probe.coef_[list(probe.classes_).index('child')],
        dtype=model.cfg.dtype,
        device=DEVICE
    )
    resid_pre[0, -1, :] += (child_direction - direction_vector) * strength
    return resid_pre

with open(promptFile, "r") as f:
    prompts = f.read().splitlines()


neutral_prompt = "HUMAN: " + prompts[0] + "\n\nASSISTANT: "
intervention_strengths = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Prepare a list to collect results
results = []

print("Generating responses for all target ages and intervention strengths...")

for target_age in age_categories:
    direction_vector = probe_directions[target_age]
    for strength in intervention_strengths:
        if strength == 0.0:
            # No intervention, no hook
            model.reset_hooks()
            response = model.generate(
                neutral_prompt,
                max_new_tokens=100,
                do_sample=True,
                temperature=temperature
            )
        else:
            hook_fn = partial(
                patch_age_direction,
                direction_vector=direction_vector,
                strength=strength
            )
            activation_name = get_activation_name(LAYER_TO_PROBE)
            model.add_hook(activation_name, hook_fn)
            response = model.generate(
                neutral_prompt,
                max_new_tokens=100,
                do_sample=True,
                temperature=temperature,
            )
            model.reset_hooks()
        results.append({
            "target_age": target_age,
            "intervention_strength": strength,
            "response": response
        })
        print(f"Target age: {target_age}, Strength: {strength}")
        print(response)
        print("-" * 50)

# Convert results to DataFrame and save
df = pd.DataFrame(results)
df.to_csv("OUTDOORACTIVITIEScausal_intervention_responses.csv", index=False)



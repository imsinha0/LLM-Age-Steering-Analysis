from transformer_lens import HookedTransformer
import torch
from tqdm.auto import tqdm
import pickle



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
numLayers = 28

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B", device=DEVICE)
model.eval()

DATA_PATH = "llama_age_1"
PROBE_DATA_SAVE_PATH = "probe_data_3b"
LAYER_TO_PROBE = [i for i in range(0, numLayers)]
ELICIATION_PROMPT = "I think the age of this user is "

def get_activation_name(layer):
    return f"blocks.{layer}.hook_resid_post"

numConversations = 150
ageCategories = ["child", "adolescent", "adult", "older adult"]

probe_data = {layer: {'activations': [], 'labels': []} for layer in LAYER_TO_PROBE}

for i in tqdm(range(numConversations)):
    for age in ageCategories:
        file_name = f"{DATA_PATH}/conversation_{i}_age_{age}.txt"
        with open(file_name, "r") as f:
            conv = f.read()
        conv = conv + "\n" +ELICIATION_PROMPT
        tokens = model.to_tokens(conv)

        _, cache = model.run_with_cache(
            tokens,
            return_type=None,
            names_filter=[get_activation_name(layer) for layer in LAYER_TO_PROBE]
        )
        for layer in LAYER_TO_PROBE:
            final_token_activation = cache[get_activation_name(layer)][0, -1, :].detach().cpu().numpy()
            probe_data[layer]['activations'].append(final_token_activation)
            probe_data[layer]['labels'].append(age)


with open(PROBE_DATA_SAVE_PATH, "wb") as f:
    pickle.dump(probe_data, f)


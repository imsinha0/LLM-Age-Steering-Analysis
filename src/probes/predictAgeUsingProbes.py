'''
import pickle
import numpy as np
from transformer_lens import HookedTransformer
import torch
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def get_activation_name(layer):
    return f"blocks.{layer}.hook_resid_post"

#load all the probes
probes = {}
scalers = {}
for i in range(28):
    data = pickle.load(open(f"trained_probes3b/probe_trained_age_{i}.pkl", "rb"))
    probes[i] = data['probe']
    scalers[i] = data['scaler']

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B", device=DEVICE)

ageCategories = ["adolescent", "adult", "child", "older adult"]

ELICIATION_PROMPT = "I think the age of this user is "


#run model with some prompt and see the activations at each layer
'''
file_name = "llama_age_1/conversation_0_age_adolescent.txt"
with open(file_name, "r") as f:
    conv = f.read()
'''

conv = "USER: I like dinosaurs and eating candy"

conv = conv + "\n" +ELICIATION_PROMPT

tokens = model.to_tokens(conv)
#tokens = model.to_tokens(prompt)

_, cache = model.run_with_cache(
    tokens,
    return_type=None,
    names_filter=[get_activation_name(layer) for layer in range(28)]
)

for layer in range(28):
    activation = cache[get_activation_name(layer)][0, -1, :]
    # Convert from CUDA tensor to numpy array
    activation = activation.detach().cpu().numpy()
    activation_2d = np.array([activation])
    activation_scaled = scalers[layer].transform(activation_2d)
    probabilities = probes[layer].predict_proba(activation_scaled)[0]
    
    # Get the actual class names from the trained model
    class_labels = probes[layer].classes_
    predicted_idx = probabilities.argmax()
    predicted_class = class_labels[predicted_idx]
    
    print(f"Layer {layer}: Probabilities: {probabilities}")
    print(f"Layer {layer}: Predicted index: {predicted_idx}")
    print(f"Layer {layer}: Predicted class: {predicted_class}")
    print()


with open("probe_data_3b", "rb") as f:
    probe_data = pickle.load(f)

# Let's debug by checking the first few samples
for layer in range(3):  # Just check first 3 layers for debugging
    print(f"\n=== Layer {layer} ===")
    
    # Get the full dataset for this layer
    X = probe_data[layer]['activations']
    y = probe_data[layer]['labels']
    
    # Apply the same train/test split as in training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test on the first few samples from test set
    for sample_idx in range(min(3, len(X_test))):
        activation = X_test[sample_idx]
        true_label = y_test[sample_idx]
        
        # Check data types and shapes
        print(f"Sample {sample_idx}:")
        print(f"  Activation shape: {activation.shape}, type: {type(activation)}")
        print(f"  True label: {true_label}")
        
        # Convert to numpy array if needed
        if hasattr(activation, 'detach'):
            activation = activation.detach().cpu().numpy()
        
        activation_2d = np.array([activation])
        
        # Apply scaler
        activation_scaled = scalers[layer].transform(activation_2d)
        
        # Get prediction
        prediction = probes[layer].predict(activation_scaled)[0]
        probabilities = probes[layer].predict_proba(activation_scaled)[0]
        
        # Get the class labels in the order they appear in probabilities
        class_labels = probes[layer].classes_
        print(f"  Class order: {class_labels}")
        print(f"  Probabilities: {probabilities}")
        
        # Show probability for each class
        for i, (class_label, prob) in enumerate(zip(class_labels, probabilities)):
            print(f"    {class_label}: {prob:.4f}")
        
        print(f"  Predicted: (index: {prediction})")
        print(f"  Correct: {prediction == true_label}")
        print()
'''

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
import torch
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def get_activation_name(layer):
    return f"blocks.{layer}.hook_resid_post"

# load all the probes
probes = {}
scalers = {}
for i in range(28):
    data = pickle.load(open(f"trained_probes3b/probe_trained_age_{i}.pkl", "rb"))
    probes[i] = data['probe']
    scalers[i] = data['scaler']

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B", device=DEVICE)

ageCategories = ["adolescent", "adult", "child", "older adult"]

ELICIATION_PROMPT = "I think the age of this user is "

conv = "USER: Suggest fun things to do during retirement"
conv = conv + "\n" + ELICIATION_PROMPT

tokens = model.to_tokens(conv)

_, cache = model.run_with_cache(
    tokens,
    return_type=None,
    names_filter=[get_activation_name(layer) for layer in range(28)]
)

# Collect probabilities into a DataFrame
probs_matrix = []

for layer in range(28):
    activation = cache[get_activation_name(layer)][0, -1, :]
    activation = activation.detach().cpu().numpy()
    activation_2d = np.array([activation])
    activation_scaled = scalers[layer].transform(activation_2d)
    probabilities = probes[layer].predict_proba(activation_scaled)[0]
    
    probs_matrix.append(probabilities)

df = pd.DataFrame(probs_matrix, columns=ageCategories, index=[f"Layer {i}" for i in range(28)])

# Plot heatmap table
plt.figure(figsize=(10, 12))
plt.title("Age Prediction Probabilities Across Layers for prompt: Suggest fun things to do during retirement", fontsize=10, pad=20)

# use imshow-style heatmap
plt.imshow(df.values, aspect="auto", cmap="YlGnBu")

# add labels
plt.xticks(np.arange(len(ageCategories)), ageCategories, rotation=30, ha="right")
plt.yticks(np.arange(28), df.index)

# annotate cells with probabilities
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        plt.text(j, i, f"{df.iloc[i, j]:.2f}", ha="center", va="center", color="black")

plt.colorbar(label="Probability")
plt.tight_layout()
plt.savefig("graphMaker/exp1/probeProbabilitiesOlder Adult.png")

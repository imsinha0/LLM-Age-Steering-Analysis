
import pickle
import matplotlib.pyplot as plt

#data = pickle.load(open("../../probe_accuracy_results3b.pkl", "rb")) #this is a dictionary mapping layer to accuracy
data = pickle.load(open("../../probe_accuracy_results3bDetailed.pkl", "rb"))

# Collect all labels
all_labels = set()
for layer_dict in data.values():
    all_labels.update(layer_dict.keys())
all_labels = sorted(all_labels)

# Plot
plt.figure(figsize=(10, 6))
for label in all_labels:
    layers = sorted(data.keys())
    accuracies = [data[layer].get(label, None) for layer in layers]
    plt.plot(layers, accuracies, label=label)

plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.title("Probe Accuracy per Layer by Label")
plt.legend()
plt.grid(True)
plt.ylim(bottom=0.7)  # Ensure y-axis starts at 0
plt.tight_layout()
plt.savefig("probeAccuracyDetailed.png")

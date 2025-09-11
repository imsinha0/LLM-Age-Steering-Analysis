
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

numLayers = 28
with open("probe_data_3b", "rb") as f:
    probe_data = pickle.load(f)

LAYERS_TO_TRAIN = [i for i in range(0, numLayers)]

# Store overall accuracy and per-label accuracy for each layer
accuracy_results = {}
per_label_accuracy_results = {}

for layer in LAYERS_TO_TRAIN:
    X = probe_data[layer]['activations'] #each of the 600 activations has size 3072
    y = probe_data[layer]['labels'] #each of the 600 labels is a string

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    probe = LogisticRegression(random_state=42, max_iter=1000)
    probe.fit(X_train, y_train)
    
    accuracy = probe.score(X_test, y_test)
    print(f"Layer {layer} overall accuracy: {accuracy}")
    accuracy_results[layer] = accuracy

    # Per-label accuracy
    y_pred = probe.predict(X_test)
    labels = sorted(list(set(y_test)))
    per_label_acc = {}
    for label in labels:
        idxs = [i for i, true_label in enumerate(y_test) if true_label == label]
        if idxs:
            correct = sum(1 for i in idxs if y_pred[i] == label)
            per_label_acc[label] = correct / len(idxs)
        else:
            per_label_acc[label] = None  # No samples for this label in test set
    print(f"Layer {layer} per-label accuracy: {per_label_acc}")
    per_label_accuracy_results[layer] = per_label_acc
    
    with open(f"trained_probes3b/probe_trained_age_{layer}.pkl", "wb") as f:
        pickle.dump({'probe': probe, 'scaler': scaler}, f)

# Save accuracy results to a file
with open("probe_accuracy_results3b.pkl", "wb") as f:
    pickle.dump(accuracy_results, f)

with open("probe_accuracy_results3bDetailed.pkl", "wb") as f:
    pickle.dump(per_label_accuracy_results, f)


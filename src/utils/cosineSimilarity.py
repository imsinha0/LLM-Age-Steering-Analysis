import numpy as np
import pickle
import matplotlib.pyplot as plt

ageCategories = ["child", "adolescent", "adult", "older adult"]

data = pickle.load(open("trained_probes3b/probe_trained_age_16.pkl", "rb"))
probe_trained_age_20 = data['probe']

# Get the probe directions for each age
probe_directions = {}
for age in ageCategories:
    probe_directions[age] = probe_trained_age_20.coef_[list(probe_trained_age_20.classes_).index(age)]

# Compute cosine similarity matrix
cosine_sim_matrix = np.zeros((len(ageCategories), len(ageCategories)))
for i, age1 in enumerate(ageCategories):
    for j, age2 in enumerate(ageCategories):
        v1 = probe_directions[age1]
        v2 = probe_directions[age2]
        cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosine_sim_matrix[i, j] = cosine_sim

# Print as a table
print("Cosine Similarity Table:")
header = " " * 15 + "".join([f"{age:>15}" for age in ageCategories])
print(header)
for i, age1 in enumerate(ageCategories):
    row = f"{age1:>15}"
    for j in range(len(ageCategories)):
        row += f"{cosine_sim_matrix[i, j]:15.3f}"
    print(row)

# Plot the cosine similarity matrix as a heatmap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cosine_sim_matrix, cmap='viridis', vmin=-1, vmax=1)

# Show all ticks and label them
ax.set_xticks(np.arange(len(ageCategories)))
ax.set_yticks(np.arange(len(ageCategories)))
ax.set_xticklabels(ageCategories)
ax.set_yticklabels(ageCategories)

# Rotate the tick labels and set alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(ageCategories)):
    for j in range(len(ageCategories)):
        text = ax.text(j, i, f"{cosine_sim_matrix[i, j]:.2f}",
                       ha="center", va="center", color="w" if abs(cosine_sim_matrix[i, j]) < 0.5 else "black")

ax.set_title("Cosine Similarity Between Probe Directions")
fig.colorbar(im, ax=ax, label="Cosine Similarity")
plt.tight_layout()
plt.savefig("cosine_similarity_heatmap.png")
plt.close()

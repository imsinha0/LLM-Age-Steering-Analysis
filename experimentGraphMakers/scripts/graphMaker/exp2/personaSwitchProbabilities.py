import pandas as pd
import matplotlib.pyplot as plt

dataNormal1 = pd.read_csv("../../personaSwitchDSProbabilities/conversation_child_adult_0.txt_probs.csv")
3

dataNormal2 = pd.read_csv("../../personaSwitchDSProbabilities/conversation_child_adult_1.txt_probs.csv")
dataSecret1 = pd.read_csv("../../personaSwitchDSSecretProbabilities/conversation_child_adult_0.txt_probs.csv")
dataSecret2 = pd.read_csv("../../personaSwitchDSSecretProbabilities/conversation_child_adult_1.txt_probs.csv")

# Select the first 7 rows for each DataFrame
dfs = [
    dataNormal1.head(7),
    dataNormal2.head(7),
    dataSecret1.head(7),
    dataSecret2.head(7)
]
titles = [
    "Explicit 1",
    "Explicit 2",
    "Implicit 1",
    "Implicit 2"
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Add a main title for the whole figure
fig.suptitle("Persona Switch Probabilities Across Conditions", fontsize=18, y=1.02)

ims = []
for idx, (df, title) in enumerate(zip(dfs, titles)):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    im = ax.imshow(df.values, aspect="auto", cmap="YlGnBu")
    ims.append(im)
    ax.set_title(title)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"Row {j+1}" for j in range(len(df))])
    # Annotate cells
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            ax.text(c, r, f"{df.iloc[r, c]:.2f}", ha="center", va="center", color="black")

# Add a single colorbar for all plots, placing it further to the right of the entire 2x2 grid

# Use the first image for the colorbar, and place it to the right of the whole figure
# Increase the 'pad' value to move the colorbar further right, and adjust 'right' in subplots_adjust
cbar = fig.colorbar(ims[0], ax=axes, orientation='vertical', fraction=0.025, pad=0.15)
cbar.set_label("Probability")

plt.subplots_adjust(top=0.92, wspace=0.25, hspace=0.25, right=0.82)
plt.savefig("allPersonaSwitchProbabilities.png", bbox_inches='tight')


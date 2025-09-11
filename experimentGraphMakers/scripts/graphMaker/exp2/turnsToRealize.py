import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data1 = pd.read_csv("../../SECRETprobe_obstinancy_summary.csv")
data2 = pd.read_csv("../../probe_obstinancy_summary.csv")

# Ensure the categories are in the same order for both datasets
categories = list(data1["switched_to_user"])
x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 5))

rects1 = ax.bar(x - width/2, data1["average_turns_after_switch"], width, label="Switch without Telling")
rects2 = ax.bar(x + width/2, data2["average_turns_after_switch"], width, label="Switch with Telling")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Turns After Switch')
ax.set_xlabel('Switched To User')
ax.set_title('Average Turns After Switch by User (Switch without Telling vs Switch with Telling)')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

fig.tight_layout()
plt.savefig("turnsToRealize.png")
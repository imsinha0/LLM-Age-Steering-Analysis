import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../../model_meta_knowledge_direct_accuracy_by_age.csv")

plt.figure(figsize=(10, 6))
plt.bar(data['age_group'], data['accuracy'], color='skyblue')
plt.xlabel("Age Group")
plt.ylabel("Accuracy")
plt.title("Accuracy of Model's Meta Knowledge by Age Group")
plt.ylim(0, 1)
plt.savefig("direct_accuracy.png")
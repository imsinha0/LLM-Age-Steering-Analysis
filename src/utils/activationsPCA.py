from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
import numpy as np

data = pickle.load(open("probe_data_3b", "rb"))
data = data[20]

activations = np.vstack(data['activations'])

labels = data['labels']

# Encode in the order: child, adolescent, adult, older adult
ordered_labels = ["child", "adolescent", "adult", "older adult"]
label_to_num = {label: i for i, label in enumerate(ordered_labels)}
numeric_labels = [label_to_num[label] for label in labels]

pca = PCA(n_components=2)
activations_2d = pca.fit_transform(activations)

scatter = plt.scatter(activations_2d[:, 0], activations_2d[:, 1], c=numeric_labels, cmap='viridis')
cbar = plt.colorbar(scatter, label='Age Group (encoded)')
# Add tick labels to colorbar to show which number corresponds to which age group
cbar.set_ticks(list(label_to_num.values()))
cbar.set_ticklabels([f"{num}: {label}" for label, num in label_to_num.items()])

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Neural Activations by Age Group', fontsize=14, pad=15)
# Add a subtitle below the plot for the encoding legend
plt.suptitle('Age group encoding: ' + ', '.join([f"{num}={label}" for label, num in label_to_num.items()]), fontsize=10, y=0.92)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('activationsPCA.png')

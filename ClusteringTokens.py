import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load aggregated token embeddings
with open("product_aggregated_token_embeddings.json", "r") as f:
    product_embeddings = json.load(f)

# Convert dictionary values to NumPy array
product_ids = list(product_embeddings.keys())
embeddings_array = np.array(list(product_embeddings.values()))

# Define the number of clusters (Tune this as needed)
num_clusters = 6  # Adjust based on your dataset size

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings_array)

# Create a mapping of product_id to cluster label
cluster_map = {product_ids[i]: int(labels[i]) for i in range(len(product_ids))}
print(cluster_map["1"])

# Save cluster assignments to a file
with open("product_clusters_token.json", "w") as f:
    json.dump(cluster_map, f)

# Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings_array)

# Scatter plot for visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis", alpha=0.6)
plt.colorbar(scatter, label="Cluster Label")
plt.title("Token - Product Clustering using K-Means (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

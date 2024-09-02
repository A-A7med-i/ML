from sklearn.decomposition import PCA
import numpy as np

np.random.seed(42)

# Generate random data
X = np.random.rand(200, 5)

# Create PCA object
pca = PCA()

# Fit PCA to the data (training)
X_PCA = pca.fit_transform(X)


# Print explained variance ratio
print(pca.explained_variance_ratio_)

# Print feature loadings (optional)
print("Feature loadings:")
print(pca.components_)

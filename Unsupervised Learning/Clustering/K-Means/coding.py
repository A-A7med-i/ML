from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = load_iris()

X = df.data
y = df.target


kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(X)

y_pred = kmeans.labels_


inertia = kmeans.inertia_
silhouette_avg = silhouette_score(X, y_pred)


print("Silhouette score:", silhouette_avg)
print("Inertia:", inertia)
library(cluster)

# Load Iris dataset
data(iris)

# Perform K-means clustering
kmeans_results <- kmeans(iris[, 1:4], centers = 3, nstart = 25)

# Extract cluster labels and total within-cluster sum of squares
cluster_labels <- kmeans_results$cluster
total_withinss <- kmeans_results$tot.withinss 

# Calculate distances
dist_matrix <- dist(iris[, 1:4])

# Calculate silhouette score
silhouette_avg <- silhouette(cluster_labels, dist_matrix)

# Print results
cat("Average Silhouette Width:", mean(silhouette_avg[, 3]), "\n")
cat("Total Within-Cluster Sum of Squares:", total_withinss, "\n")


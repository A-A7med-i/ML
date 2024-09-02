set.seed(42)

X <- matrix(runif(1000), nrow = 200, ncol = 5)

pca_result <- princomp(X)

print("Explained Variance Ratio:")
print(pca_result$sdev^2 / sum(pca_result$sdev^2))


print("Feature loadings:")
print(pca_result$loading)

X_pca <- predict(pca_result)

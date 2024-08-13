# Load required libraries
library(glmnet)
library(MASS)

# Load the diabetes dataset
data(diabetes, package="lars")

X <- diabetes$x
y <- diabetes$y


# Split the data into training and testing sets
set.seed(42)

train_indices <- sample(1:nrow(X), 0.8 * nrow(X))
X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

# Fit Lasso model
lasso_model <- cv.glmnet(X_train, y_train, alpha = 1)
best_lambda_lasso <- lasso_model$lambda.min


# Fit Ridge model
ridge_model <- cv.glmnet(X_train, y_train, alpha = 0)
best_lambda_ridge <- ridge_model$lambda.min


# Make predictions
lasso_pred <- predict(lasso_model, s = best_lambda_lasso, newx = X_test)
ridge_pred <- predict(ridge_model, s = best_lambda_ridge, newx = X_test)


# Calculate MSE
lasso_mse <- mean((y_test - lasso_pred)^2)
ridge_mse <- mean((y_test - ridge_pred)^2)


# Print results
cat("Lasso MSE:", lasso_mse, "\n")
cat("Ridge MSE:", ridge_mse, "\n")


# Print coefficients
cat("\nLasso Coefficients:\n")
print(coef(lasso_model, s = best_lambda_lasso))

cat("\nRidge Coefficients:\n")
print(coef(ridge_model, s = best_lambda_ridge))




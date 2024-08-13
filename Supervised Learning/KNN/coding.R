# load required packages
library(caret)
library(mlbench)

# Set seed for reproducibility
set.seed(42)

# Regression task
data(BostonHousing)
X <- BostonHousing[, -14]  # All columns except the last one
y <- BostonHousing$medv    # The last column (target)


# Create train/test split
train_index <- sample(1:nrow(X), 0.8 * nrow(X))
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]


# Train KNN regression model
knn_reg_model <- train(X_train, y_train,
                       method = "knn",
                       tuneGrid = data.frame(k = 1:19),
                       trControl = trainControl(method = "cv", number = 5))


# Print results
cat("Regression Results:\n")
print(knn_reg_model$bestTune$k)
cat("\nBest k value:", knn_reg_model$bestTune$k, "\n")


# Make predictions and calculate metrics
y_pred <- predict(knn_reg_model, newdata = X_test)
mse <- mean((y_test - y_pred) ^ 2)
r2 <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)


cat("Final Model Performance:\n")
cat("MSE:", round(mse, 2), "\n")
cat("R-squared:", round(r2, 2), "\n\n")


# Classification task
data("iris")

X <- iris[, -5]  # All columns except the last one
y <- iris[, 5]   # The last column (target)


# Create train/test split
train_index <- sample(1:nrow(X), 0.8 * nrow(X))
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Train KNN classification model
knn_class_model <- train(X_train, y_train,
                         method = "knn",
                         tuneGrid = data.frame(k = 1:19),
                         trControl = trainControl(method = "cv", number = 5))


# Print results
cat("Classification Results:\n")
print(knn_class_model)
cat("\nBest k value:", knn_class_model$bestTune$k, "\n")


# Make predictions and calculate accuracy
y_pred <- predict(knn_class_model, newdata = X_test)


# Print classification report
conf_matrix <- confusionMatrix(y_pred, y_test)
print(conf_matrix)

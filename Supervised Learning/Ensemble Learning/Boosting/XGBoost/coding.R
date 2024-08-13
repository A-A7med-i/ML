library(xgboost)
library(mlbench)

# Regression
data(BostonHousing)

X <- as.matrix(sapply(BostonHousing[, -14], as.numeric))
y <- BostonHousing$medv

model_reg <- xgboost(data = X, label = y, nrounds = 100, objective = "reg:squarederror")

pred_reg <- predict(model_reg, X)

print(paste("MSE:", mean((y - pred_reg)^2)))

# Classification


# Prepare the data
data(BreastCancer)

complete_data <- na.omit(BreastCancer[, -1])  # Remove ID column and NA values

# Create X matrix and y vector
X <- as.matrix(model.matrix(~ . - Class - 1, data = complete_data))
y <- as.numeric(complete_data$Class == "malignant")


model_clf <- xgboost(data = X, label = y, nrounds = 100, objective = "binary:logistic")

pred_clf <- predict(model_clf, X) > 0.5

print(paste("Accuracy:", mean(pred_clf == y)))


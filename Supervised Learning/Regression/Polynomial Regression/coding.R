library(MASS)

data(Boston)


colnames(Boston)

set.seed(42)

train_indices <- sample(1:nrow(Boston), 0.8 * nrow(Boston))
train_data <- Boston[train_indices,]
test_data <- Boston[-train_indices,]

model <- lm(medv ~ poly(lstat, degree = 2), data = train_data)

prediction <- predict(model, newdata = test_data)

r_squared <- summary(model)$r.squared
mse <- mean((test_data$medv - prediction) ^ 2)
mae <- mean(abs(test_data$medv - prediction))
rmse <- sqrt(mse)

cat("Mean Squared Error:", mse, "\n")
cat("Root Mean Squared Error:", rmse, "\n")
cat("Mean Absolute Error:", mae, "\n")
cat("R-squared:", r_squared, "\n")


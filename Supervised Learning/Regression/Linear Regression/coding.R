data("mtcars")

colnames(mtcars)

set.seed(42)

train_indices <- sample(1:nrow(mtcars), 0.8 * nrow(mtcars))
train_data <- mtcars[train_indices, ]
test_data <- mtcars[-train_indices, ]

model <- lm(mpg ~ ., data = train_data)

prediction <- predict(model, newdata = test_data)

r_squared <- summary(model)$r.squared
mse <- mean((test_data$mpg - prediction) ^ 2)
mae <- mean(abs(test_data$mpg - prediction))
rmse <- sqrt(mse)

print(paste("Mean Squared Error:", mse))
print(paste("Root Mean Squared Error:", rmse))
print(paste("Mean Absolute Error:", mae))
print(paste("R-squared:", r_squared))
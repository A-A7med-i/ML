# Load necessary libraries
library(e1071)
library(caret)
library(datasets)



# Regression task using mtcars dataset
data(mtcars)

# Split the data
set.seed(42)

sample <- sample(1:nrow(mtcars), 0.8 * nrow(mtcars))
train <- mtcars[sample, ]
test <- mtcars[-sample, ]

# Create and train the SVR model
reg_model <- svm(mpg ~ ., data = train, kernel = "radial", cost = 3, epsilon = 0.1)

# Make predictions
y_pred <- predict(reg_model, newdata = test)


# Calculate metrics
mse <- mean((test$mpg - y_pred) ^ 2)
r2 <- 1 - sum((test$mpg - y_pred)^2) / sum((test$mpg - mean(test$mpg))^2)

cat("Regression Results:\n")
cat("Mean Squared Error:", mse, "\n")
cat("R-squared:", r2, "\n\n")




# Or using train method

svm_model <- train(mpg ~ ., 
                   data = train,
                   method = "svmRadial",
                   trControl = trainControl(method = "cv", number = 5),
                   tuneLength = 10)

print(svm_model$bestTune)






# Classification task using iris dataset
data(iris)

# Split the data
set.seed(42)
sample <- sample(1:nrow(iris), 0.8 * nrow(iris))
train <- iris[sample, ]
test <- iris[-sample, ]


# Create and train the SVC model
class_model <- svm(Species ~ ., data = train, kernel = "polynomial", degree = 3, cost = 1)

# Make predictions
y_pred <- predict(class_model, test)


# Print classification report
conf_matrix <- confusionMatrix(test$Species, y_pred)

print(conf_matrix)


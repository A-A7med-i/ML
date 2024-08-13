# Load required libraries
library(randomForest)
library(caret)
library(ipred)
library(datasets)


# Load datasets
data(mtcars)
data(iris)

# Function to split data
split_data <- function(X, y, seed = 42){
  set.seed(seed)
  sample <- sample(1:nrow(X), 0.8 * nrow(X))
  
  X_train <- X[sample, ]
  X_test <- X[-sample, ]
  y_train <- y[sample]
  y_test <- y[-sample]
  
  return(list(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test))
}

# Regression

# Prepare data
X <- mtcars[, -1]
y <- mtcars$mpg


# Split data
split <- split_data(X, y)


# Build random forest model
random_reg_model <- randomForest(x = split$X_train, y = split$y_train, ntree = 500)


# Make predictions
y_pred <- predict(random_reg_model, newdata = split$X_test)


# Calculate R-squared
cat("R2 Score: ", round(R2(y_pred, split$y_test), 2))


# Or
bagging_model <- bagging(
  formula = y ~ .,
  data = data.frame(y = split$y_train, split$X_train),
  nbagg = 500
)


# Make predictions with bagging model
y_pred_bagging <- predict(bagging_model, newdata = split$X_test)



# classification


# Prepare data
X <- iris[, -5]
y <- iris$Species


# Split data
split <- split_data(X, y)

# Build random forest model
random_class_model <- randomForest(x = split$X_train, y = split$y_train, ntree = 500)


# Make predictions
y_pred <- predict(random_class_model, newdata = split$X_test)


# Make confusion Matrix
confusion_Matrix <- confusionMatrix(y_pred, split$y_test)
print(confusion_Matrix)


library(caret)
library(nnet)




# Binary classification

data(BreastCancer, package = "mlbench")
set.seed(42)

X <- BreastCancer[,-1]
y <- as.numeric(BreastCancer$Class) - 1

train_index <- sample(1:nrow(BreastCancer), 0.8 * nrow(BreastCancer))
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

model <- glm(y_train ~ ., data = as.data.frame(X_train), family = "binomial")

prediction <- predict(model, newdata = as.data.frame(X_test), type = "response")
prediction <- ifelse(prediction > 0.5, 1, 0)

conf_matrix <- confusionMatrix(factor(prediction), factor(y_test))

accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Recall"]
f1_score <- conf_matrix$byClass["F1"]

cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")

# Multiclass classification
data(iris)

set.seed(42)

X <- iris[,-5]
y <- iris$Species


train_index <- sample(1:nrow(iris), 0.8 * nrow(iris))
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

model <- multinom(y_train ~ ., data = as.data.frame(X_train))

predictions <- predict(model, newdata = as.data.frame(X_test), type = "class")

predictions <- factor(predictions, levels = levels(y))

y_test <- factor(y_test, levels = levels(y))

conf_matrix <- confusionMatrix(predictions, y_test)

accuracy <- conf_matrix$overall["Accuracy"]
precision <- mean(conf_matrix$byClass[, "Precision"])
recall <- mean(conf_matrix$byClass[, "Recall"])
f1_score <- mean(conf_matrix$byClass[, "F1"])

cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
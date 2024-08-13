library(MASS)
library(caret)
library(gbm)

data(Boston)
X <- Boston[, -14]  # Exclude the target variable 'medv', keep as data frame
y <- Boston$medv


train_indices <- sample(1:nrow(X), 0.8 * nrow(X))

X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

gbr <- gbm(
  formula = y_train ~ .,
  data = cbind(X_train, y_train),
  distribution = "gaussian",
  n.trees = 300
)

y_pred <- predict(gbr, newdata = X_test, ntree = 300)

r2 <- R2(y_test, y_pred)
print(paste("R2:", round(r2, 2)))



data(BreastCancer)
X <- BreastCancer[, -1]  # Exclude the ID column
y <- as.numeric(BreastCancer$Class) - 1

train_indices <- sample(1:nrow(X), 0.8 * nrow(X))

X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

y_test <- factor(y_test, levels = c(0, 1))


gbc <- gbm(
  formula = y_train ~ .,
  data = cbind(X_train, y_train),
  distribution = "bernoulli",
  n.trees = 300
)

y_pred <- predict(gbc, newdata = X_test, ntree = 300)
y_pred <- factor(ifelse(y_pred > 0.5, 1, 0), levels = c(0, 1))


confusion_Matrix = confusionMatrix(y_test, y_pred)

print(confusion_Matrix)



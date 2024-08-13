library(rpart)
library(caret)

data(mtcars)


# Regression

set.seed(42)

sample <- sample(1:nrow(mtcars), 0.8 * nrow(mtcars))
train <- mtcars[sample, ]
test <- mtcars[-sample, ]


reg_model <- rpart(mpg ~ ., data = train)
y_pred <- predict(reg_model, newdata = test)


mse <- mean((test$mpg - y_pred) ^ 2)
r2 <- 1 - sum((test$mpg - y_pred)^2) / sum((test$mpg - mean(test$mpg))^2)

print(paste("MSE:", mse))
print(paste("R-squared:", r2))


# Or using train method

reg_dt_model <- train(mpg ~ ., 
                   data = train,
                   method = "rpart",
                   trControl = trainControl(method = "cv", number = 5),
                   tuneLength = 10)

print(reg_dt_model)

print(reg_dt_model$bestTune)



# Classification

data("iris")

set.seed(42)
sample <- sample(1:nrow(iris), 0.8 * nrow(iris))
train <- iris[sample, ]
test <- iris[-sample, ]

class_model <- rpart(Species ~ ., data = train)
y_pred <- predict(class_model, test, type = "class")

conf_matrix <- confusionMatrix(test$Species, y_pred)

print(conf_matrix)





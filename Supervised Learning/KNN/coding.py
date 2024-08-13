import numpy as np
from sklearn.datasets import load_diabetes, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    classification_report,
)


# regression
df = load_diabetes()

X, y = df.data, df.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

n_neighbors = np.arange(1, 10)

mses = []
r2_errors = []

for k in n_neighbors:
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    mse_error = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mses.append(mse_error)
    r2_errors.append(r2)


best_k = n_neighbors[np.argmin(mses)]
print(f"Best k value: {best_k}")


best_model = KNeighborsRegressor(n_neighbors=best_k)
best_model.fit(X_train, y_train)


y_pred_final = best_model.predict(X_test)
final_mse = mean_squared_error(y_test, y_pred_final)
final_r2 = r2_score(y_test, y_pred_final)

print(f"Final Model Performance:")
print(f"MSE: {final_mse:.2f}")
print(f"R-squared: {final_r2:.2f}")


# classification

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_classes=3, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

accuracy_scores = []

for k in n_neighbors:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)


best_k = n_neighbors[np.argmax(accuracy_scores)]
print(f"Best k value: {best_k}")


best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)

y_pred_final = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)

print("Final Model Performance:")
print(f"Accuracy: {final_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the diabetes dataset
df = load_diabetes()
X, y = df.data, df.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Define a range of alpha values for Lasso and Ridge
alphas = np.logspace(-4, 0, 50)

# Lists to store the results
lasso_mses = []
ridge_mses = []
lasso_r2s = []
ridge_r2s = []

# Train and evaluate Lasso and Ridge for different alpha values
for alpha in alphas:
    # Lasso
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    lasso_mses.append(mean_squared_error(y_test, y_pred_lasso))
    lasso_r2s.append(r2_score(y_test, y_pred_lasso))

    # Ridge
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    ridge_mses.append(mean_squared_error(y_test, y_pred_ridge))
    ridge_r2s.append(r2_score(y_test, y_pred_ridge))

# Train a regular Linear Regression model for comparison
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

# Find the best alpha for Lasso and Ridge
best_alpha_lasso = alphas[np.argmin(lasso_mses)]
best_alpha_ridge = alphas[np.argmin(ridge_mses)]

print(f"Best alpha for Lasso: {best_alpha_lasso:.4f}")
print(f"Best alpha for Ridge: {best_alpha_ridge:.4f}")

# Train final models with best alphas
final_lasso = Lasso(alpha=best_alpha_lasso, random_state=42)
final_lasso.fit(X_train, y_train)

final_ridge = Ridge(alpha=best_alpha_ridge, random_state=42)
final_ridge.fit(X_train, y_train)

# Print feature importances for Lasso
print("\nLasso Feature Importances:")
for feature, importance in zip(df.feature_names, final_lasso.coef_):
    print(f"{feature}: {importance:.4f}")

# Print feature importances for Ridge
print("\nRidge Feature Importances:")
for feature, importance in zip(df.feature_names, final_ridge.coef_):
    print(f"{feature}: {importance:.4f}")

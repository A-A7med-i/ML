from sklearn.datasets import load_digits, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score

# Regression

X, y = load_diabetes(return_X_y=True)


train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.25, random_state=42
)


gbr = GradientBoostingRegressor(
    loss="absolute_error",
    learning_rate=0.1,
    n_estimators=300,
    max_depth=1,
    random_state=42,
    max_features=5,
)
gbr.fit(train_X, train_y)
pred_y = gbr.predict(test_X)


r2 = r2_score(test_y, pred_y)
print("R2: {:.2f}".format(r2))


# Classification

X, y = load_digits(return_X_y=True)

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.25, random_state=42
)

gbc = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, random_state=100, max_features=5
)
gbc.fit(train_X, train_y)
pred_y = gbc.predict(test_X)


acc = accuracy_score(test_y, pred_y)
print(f"Accuracy: {acc:.2f}")

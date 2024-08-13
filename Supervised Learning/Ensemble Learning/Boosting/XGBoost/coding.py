from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score

# Regression

X, y = load_diabetes(return_X_y=True)


train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.25, random_state=42
)


xgb_reg = xgb.XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.1,
    n_estimators=300,
)
xgb_reg.fit(train_X, train_y)
pred_y = xgb_reg.predict(test_X)


r2 = r2_score(test_y, pred_y)
print("R2: {:.2f}".format(r2))


# Classification


X, y = load_breast_cancer(return_X_y=True)

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.25, random_state=42
)

xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic", n_estimators=100, learning_rate=0.1
)
xgb_clf.fit(train_X, train_y)
pred_y = xgb_clf.predict(test_X)


acc = accuracy_score(test_y, pred_y)
print(f"Accuracy: {acc:.2f}")

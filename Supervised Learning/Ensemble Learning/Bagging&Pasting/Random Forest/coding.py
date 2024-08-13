from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import precision_score, recall_score, r2_score

# Regression

df = load_breast_cancer()

X, y = df.data, df.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

random_reg_model = RandomForestRegressor(n_estimators=500, random_state=42)
random_reg_model.fit(X_train, y_train)
y_pred = random_reg_model.predict(X_test)

print("R2 Score: ", r2_score(y_test, y_pred))

for score, name in zip(random_reg_model.feature_importances_, df.feature_names):
    print(round(score, 2), name)

# Or

bagging_reg_model = BaggingRegressor(
    DecisionTreeRegressor(), n_estimators=500, random_state=42
)

bagging_reg_model.fit(X_train, y_train)
y_pred = bagging_reg_model.predict(X_test)

print("R2 Score: ", r2_score(y_test, y_pred))


# classification

df = load_iris()

X, y = df.data, df.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

random_class_model = RandomForestClassifier(n_estimators=400, random_state=42)
random_class_model.fit(X_train, y_train)
y_pred = random_class_model.predict(X_test)

print("precision: ", precision_score(y_test, y_pred, average="weighted"))
print("recall: ", recall_score(y_test, y_pred, average="weighted"))

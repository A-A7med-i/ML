from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import r2_score, recall_score, precision_score


# Regression

df = load_breast_cancer()

X, y = df.data, df.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

adaboost_reg_model = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=1), n_estimators=100
)
adaboost_reg_model.fit(X_train, y_train)
y_pred = adaboost_reg_model.predict(X_test)

print("R2: ", r2_score(y_test, y_pred))


# Classification

df = load_iris()

X, y = df.data, df.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

adaboost_class_model = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200
)
adaboost_class_model.fit(X_train, y_train)
y_pred = adaboost_class_model.predict(X_test)

print("Recall: ", recall_score(y_test, y_pred, average="macro"))
print("precision: ", precision_score(y_test, y_pred, average="macro"))

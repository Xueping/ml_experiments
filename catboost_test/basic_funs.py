from catboost import CatBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import numpy as np


def binary_class():
    # Initialize data
    cat_features = [0, 1, 2]  # indices for cat_features
    train_data = [["a", "b", 1, 4, 5, 6], ["a", "b", 4, 5, 6, 7],
                  ["c", "d", 30, 40, 50, 60]]  # cat_features should be string or int
    train_labels = [1, 1, -1]
    test_data = [["a", "b", 2, 4, 6, 8], ["a", "d", 1, 4, 50, 60]]

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='Logloss')
    # Fit model
    model.fit(train_data, train_labels, cat_features)

    # Get predicted classes
    preds_class = model.predict(test_data)
    # Get predicted probabilities for each class
    preds_proba = model.predict_proba(test_data)
    # Get predicted RawFormulaVal
    preds_raw = model.predict(test_data, prediction_type='RawFormulaVal')
    print(preds_class)
    print(preds_proba)
    print(preds_raw)


def multi_class():
    iris = load_iris()
    x_tr, x_te, y_tr, y_te = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    # Initialize CatBoostClassifier
    model = CatBoostClassifier(iterations=100, learning_rate=1, depth=6, loss_function='MultiClass', class_count=3)

    def normal_class():
        # Fit model
        model.fit(x_tr, y_tr)

        # Get predicted classes
        preds_class = model.predict(x_te)
        print("accuracy = {}".format(accuracy_score(y_te, preds_class)))
        # Get predicted probabilities for each class
        preds_proba = model.predict_proba(x_te)
        print("log_loss = {}".format(log_loss(y_te, preds_proba)))

    # use pre-training results(baseline)
    def baseline():
        # Get baseline (only with prediction_type='RawFormulaVal')
        baseline = model.predict(x_tr, prediction_type='RawFormulaVal')
        # Fit new model
        model.fit(x_tr, y_tr, baseline=baseline)
        # Get predicted classes
        preds_class = model.predict(x_te)
        print("accuracy = {}".format(accuracy_score(y_te, preds_class)))
        # Get predicted probabilities for each class
        preds_proba = model.predict_proba(x_te)
        print("log_loss = {}".format(log_loss(y_te, preds_proba)))

    def weight():
        print("running objects weights!")
        weights = np.random.random(len(x_tr))
        model.fit(x_tr, y_tr, sample_weight=weights)
        # Get predicted classes
        preds_class = model.predict(x_te)
        print("accuracy = {}".format(accuracy_score(y_te, preds_class)))
        # Get predicted probabilities for each class
        preds_proba = model.predict_proba(x_te)
        print("log_loss = {}".format(log_loss(y_te, preds_proba)))

    def best_model():
        print("running Best Model!")
        eval_set = (x_te, y_te)
        model.fit(x_tr, y_tr, use_best_model=True, eval_set=eval_set)

        # Get predicted classes
        preds_class = model.predict(x_te)
        print("accuracy = {}".format(accuracy_score(y_te, preds_class)))
        # Get predicted probabilities for each class
        preds_proba = model.predict_proba(x_te)
        print("log_loss = {}".format(log_loss(y_te, preds_proba)))

        staged_predictions = model.staged_predict(x_te)
        print(staged_predictions)
        # It is equivalent to use predict() with `ntree_limit` in loop
        # staged_predictions = []
        # for i in range(1, model.get_tree_count() + 1):
        #     staged_predictions.append(model.predict(x_te, ntree_limit=i))

    # weight()
    best_model()


if __name__ == '__main__':
    multi_class()

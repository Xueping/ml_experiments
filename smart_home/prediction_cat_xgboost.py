import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class SmartHome:

    def __init__(self):

        self.train_df = pd.read_csv('../data/smarthome/finalDataset-201612.txt')
        # print(train_df.head())

        self.X = self.train_df.drop(['speed', 'direct', 'actual_mode', 'temp'], axis=1)
        self.XG = pd.get_dummies(data=self.X, columns=['direct1', 'direct2', 'direct3', 'mode1', 'mode2', 'mode3'])

        self.y = self.train_df.temp
        # self.y = train_df.actual_mode
        self.class_count = 5

        # self.y = train_df.speed
        # self.class_count = 6
        # self.y = train_df.direct
        # self.class_count = 3

        # print(self.X.head())
        # print(self.XG.head())

    def xgboost_prediction(self):

        train_X, test_X, train_Y, test_Y = train_test_split(self.XG, self.y, train_size=.85, random_state=42)

        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_test = xgb.DMatrix(test_X, label=test_Y)
        # setup parameters for xgboost
        param = {}
        # use softmax multi-class classification
        param['objective'] = 'multi:softmax'
        # scale weight of positive examples
        param['eta'] = 0.1
        param['max_depth'] = 6
        param['silent'] = 1
        param['nthread'] = 4
        param['num_class'] = self.class_count

        watchlist = [(xg_train, 'train'), (xg_test, 'test')]
        num_round = 5
        bst = xgb.train(param, xg_train, num_round, watchlist)

        # get prediction
        pred = bst.predict(xg_test)
        error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
        print('Test error using softmax = {:.4}'.format(error_rate))
        print('Test accuracy using softmax = {:.4}'.format(1-error_rate))

        # do the same thing again, but output probabilities
        param['objective'] = 'multi:softprob'
        bst = xgb.train(param, xg_train, num_round, watchlist)

        # Note: this convention has been changed since xgboost-unity
        # get prediction, this is in 1D array, need reshape to (ndata, nclass)
        pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], self.class_count)
        pred_label = np.argmax(pred_prob, axis=1)
        error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
        print('Test error using softprob = {:.4}'.format(error_rate))
        print('Test accuracy using softprob = {:.4}'.format(1-error_rate))

    def catboost_prediction(self):

        X_train, X_validation, y_train, y_validation = train_test_split(self.X, self.y, train_size=.85, random_state=42)
        categorical_features_indices = [6, 7, 8, 9, 10, 11]

        def best_model():
            print("running Best Model!")
            eval_set = (X_validation, y_validation)
            my_best_model = CatBoostClassifier(depth=6, loss_function='MultiClass',
                                               class_count=self.class_count, random_state=42)
            my_best_model.fit(X_train, y_train, eval_set=eval_set, use_best_model=True,
                              cat_features=categorical_features_indices)

            # Get predicted classes
            preds_class = my_best_model.predict(X_validation)
            print("best accuracy = {:.4}".format(accuracy_score(y_validation, preds_class)))
            # Get predicted probabilities for each class
            preds_proba = my_best_model.predict_proba(X_validation)
            print("best log_loss = {:.4}".format(log_loss(y_validation, preds_proba)))

            # staged_predictions = best_model.staged_predict(X_validation)
            # print(staged_predictions)


        best_model()

        def normal_class():
            print("running Normal Model!")
            # Fit model
            normal_model = CatBoostClassifier(depth=6, loss_function='MultiClass', class_count=self.class_count, random_state=42)
            normal_model.fit(X_train, y_train)

            # Get predicted classes
            preds_class = normal_model.predict(X_validation)
            print("normal accuracy = {:.4}".format(accuracy_score(y_validation, preds_class)))
            # Get predicted probabilities for each class
            preds_proba = normal_model.predict_proba(X_validation)
            print("normal log_loss = {:.4}".format(log_loss(y_validation, preds_proba)))

        normal_class()

    def sklearn_comparision(self):
        train_x, test_x, train_y, test_y = train_test_split(self.XG, self.y, train_size=.85, random_state=42)
        names = ["Nearest Neighbors","MultiClass SVM", "Linear SVM", "RBF SVM",
                 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                 "Naive Bayes", "QDA", "logistic", "Gaussian Process"]

        classifiers = [
            KNeighborsClassifier(3),
            svm.SVC(decision_function_shape='ovo'),
            svm.SVC(kernel="linear", C=0.025),
            svm.SVC(gamma=2, C=1),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_jobs=2),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
            LogisticRegression(),
            GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)]

        for name, clf in zip(names, classifiers):
            clf.fit(train_x, train_y)
            print(name+" accuracy = {:.4}".format(accuracy_score(test_y, clf.predict(test_x))))

if __name__ == '__main__':
    sh = SmartHome()
    sh.xgboost_prediction()
    sh.catboost_prediction()
    # sh.sklearn_comparision()

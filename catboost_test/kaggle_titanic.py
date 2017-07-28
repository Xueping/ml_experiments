import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
import hyperopt
from catboost import Pool, CatBoostClassifier, cv, CatboostIpythonWidget

from sklearn.model_selection import train_test_split

train_df = pd.read_csv('../data/titanic/train.csv')
test_df = pd.read_csv('../data/titanic/test.csv')

# print(train_df.head())
# print(train_df.isnull().sum(axis=0))

train_df.fillna(-999, inplace=True)

X = train_df.drop(['Survived','PassengerId'], axis=1)
y = train_df.Survived

# print(X.dtypes)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.85, random_state=1234)

categorical_features_indices = np.where(X.dtypes != np.float)[0]

def prediction():
    model = CatBoostClassifier( custom_loss=['Accuracy'], random_seed=42)

    model.fit(X_train, y_train, cat_features=categorical_features_indices,use_best_model=True, eval_set=(X_validation, y_validation),
          verbose=True,  # you can uncomment this for text output
    #     plot=True
    )


    # Build model
    cbc = CatBoostClassifier(random_seed=42).fit(X_train, y_train,cat_features=[0,1,2,6,8,9])

    test_df.fillna(-999, inplace=True)
    test_set = test_df.drop(['PassengerId'], axis=1)

    defaul_preds = cbc.predict(test_set).astype(int)
    best_preds = model.predict(test_set).astype(int)

    print(confusion_matrix(defaul_preds, best_preds))
    # pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':model.predict(test_set).astype(int)}).to_csv('Titanic_catboost.csv',index=False)

def comparision():
    model_simple = CatBoostClassifier(
        eval_metric='Accuracy',
        use_best_model=False,
        random_seed=42
    )

    model_with_earlystop = CatBoostClassifier(
        eval_metric='Accuracy',
        use_best_model=True,
        random_seed=42
    )

    model_simple.fit(
        X_train, y_train,
        cat_features=categorical_features_indices,
        eval_set=(X_validation, y_validation),verbose=True
    )

    model_with_earlystop.fit(
        X_train, y_train,
        cat_features=categorical_features_indices,
        eval_set=(X_validation, y_validation),verbose=True
    )

    print (
        'Simple model validation accuracy: {:.4}'.format(
            accuracy_score(y_validation, model_simple.predict(X_validation))
        )
    )

    print(
        'Early-stopped model validation accuracy: {:.4}'.format(
            accuracy_score(y_validation, model_with_earlystop.predict(X_validation))
        )
    )

if __name__ == '__main__':
    comparision()
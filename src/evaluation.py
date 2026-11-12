import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import  classification_report,confusion_matrix,f1_score, make_scorer, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import clone

def cross_validate_models(X_train, y_train, models, preprocessor):
    
    Kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    
    scoring = {
        'precision': make_scorer(precision_score, pos_label='yes', zero_division=0),
        'recall': make_scorer(recall_score, pos_label='yes', zero_division=0),
        'f1': make_scorer(f1_score, pos_label='yes', zero_division=0)
    }

    result = pd.DataFrame()

    for name, model in models.items():

        pipeline = Pipeline([
                ("preprocessor",clone(preprocessor)),
                ("model",clone(model))
                ])
        
        scores = cross_validate(pipeline,
                                X_train,y_train,
                                cv=Kfold,
                                scoring=scoring,
                                n_jobs=-1)


        result[name] = {
            'precision': scores['test_precision'].mean(),
            'precision_std': scores['test_precision'].std(),
            'recall': scores['test_recall'].mean(),
            'recall_std': scores['test_recall'].std(),
            'f1': scores['test_f1'].mean(),
            'f1_std': scores['test_f1'].std()
        }

    return result

def evaluate_best_model(preprocessor,model, X_train, y_train, X_test, y_test):

    pipeline = Pipeline([
                    ("preprocessor",clone(preprocessor)),
                    ("model",clone(model))
                    ])

    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)

    cr = classification_report(y_test,y_pred,
                            labels=['no', 'yes'],
                            output_dict=True,
                            zero_division=0)['yes']

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=['no', 'yes']
    )

    result = {
        "precision": cr['precision'],
        "f1": cr['f1-score'],
        "recall": cr['recall'],
        "confusion_matrix": cm
    }
    
    return result
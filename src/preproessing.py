import pandas as pd
import sklearn as skl
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def procces_features(numeric_featrues,categorical_featrues):

    numeric_pipe = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
    ])

    categorical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="constant")),
    ("encoder",OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocces = ColumnTransformer(
    transformers=[("num",numeric_pipe,numeric_featrues),
                  ("cat",categorical_pipeline,categorical_featrues)],
    )

    return preprocces
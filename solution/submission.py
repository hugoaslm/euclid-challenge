from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline


def get_model():
    return Pipeline([
        ("gradboost_sk", HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0)),
    ])

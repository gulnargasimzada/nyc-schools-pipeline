"""
Machine learning helpers used by the agents: building features and creating models.
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import BASE_FEATURES, BOROUGHS, RANDOM_STATE


def build_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Build the feature matrix for a named feature set.

    The same function is used for training and test data, so both are
    always built in exactly the same way.
    """
    X = df[BASE_FEATURES].astype(float)

    if feature_set == "base_plus_borough":
        # Borough from the DBN, one-hot encoded with a fixed category list
        # (Manhattan is the reference category and is dropped)
        borough = pd.Categorical(
            df["dbn"].str[2].map(BOROUGHS), categories=list(BOROUGHS.values())
        )
        dummies = pd.get_dummies(borough, prefix="borough", drop_first=True, dtype=float)
        dummies.index = df.index
        X = pd.concat([X, dummies], axis=1)
    elif feature_set != "base":
        raise ValueError(f"Unknown feature set: {feature_set}")

    return X


def make_model(name: str, params: dict):
    """Create an untrained scikit-learn model from its name and parameters."""
    if name == "logistic_regression":
        # Linear model: features must be on the same scale
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, **params))
    if name == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, **params)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
    raise ValueError(f"Unknown model: {name}")
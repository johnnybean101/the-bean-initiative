import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# custom transformation
class ConvertNumeric(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass 

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        mileage = np.array(X.apply(lambda x: float(x['Mileage'].split(' ')[0]) if x['Mileage'] == x['Mileage'] else x['Mileage'], axis=1))
        engine = np.array(X.apply(lambda x: float(x['Engine'].split(' ')[0]) if x['Engine'] == x['Engine'] else x['Engine'], axis=1))
        power = np.array(X.apply(lambda x: float(x['Power'].split(' ')[0]) if (x['Power'] == x['Power']) and (x['Power'].split(' ')[0] != 'null') else np.nan, axis=1))
        return np.c_[mileage, engine, power]
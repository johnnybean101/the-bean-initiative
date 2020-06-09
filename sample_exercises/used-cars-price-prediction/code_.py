import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#=====================================
# Import Data
#=====================================
df = pd.read_csv('data/train-data.csv').iloc[:, 1:]

#================================
# Split Train, Test Sets
#================================
stratshuffle = StratifiedShuffleSplit(
                n_splits=1, 
                test_size=0.2, 
                random_state=42
                )
for train_indices, test_indices in stratshuffle.split(df, df['Year']):
    train = df.iloc[train_indices, :]
    test = df.iloc[test_indices, :]

#=====================================
# Data Exploration
#=====================================
train_eda = train.copy()

# check for null values, column types 
train_eda.info(verbose=True)

# check categorical column values and distributions
print('Name unique values: ' + str(len(train_eda.Name.unique()))) # too many unique values 
train_eda.Location.hist(bins=len(train_eda.Location.unique()), figsize=(15, 10), width=0.5) # distribution of where cars are sold
train_eda.Fuel_Type.value_counts() / train_eda.shape[0] # 98.87% of cars uses diesel or petrol
train_eda.Transmission.value_counts() # most cars sold are manual
train_eda.Owner_Type.value_counts() / train_eda.shape[0] # more than 80% of cars are first owned

# will need to convert the following into numerical values
pd.cut(train_eda[['Mileage']].dropna().apply(lambda x: float(x.Mileage.split(' ')[0]), axis=1), bins=5).value_counts() # most cars sold have mileage between 13 and 26
pd.cut(train_eda[['Engine']].dropna().apply(lambda x: float(x.Engine.split(' ')[0]), axis=1), bins=20).value_counts() # most cars have Engine < 1257 CC
pd.cut(
    [float(power) for power in train_eda[['Power']].dropna().apply(lambda x: x.Power.split(' ')[0], axis=1) if power != 'null'],
     bins=5).value_counts() # most cars have below 4425 BHP power

# check for numerical column values and distributions
train_eda.describe()
train_eda.Year.hist(bins=len(train_eda.Year.unique())) # most cars being sold are relatively new models
pd.cut(train_eda.Kilometers_Driven, bins=[0, 50000, 100000, 150000, 200000, np.inf]).value_counts() # most cars have around less than 10k kilometers driven
train_eda.Seats.dropna().value_counts() # most cars have 5 seats
len(train_eda.New_Price.dropna()) / train_eda.shape[0] # too many null values, more than 80% are null

# explore target column
train_eda.Price.hist(bins=200) # target distribution appears to be tail-heavy

# explore correlation between target and numerical features
train_eda.corr()['Price']

#=====================================
# Data Preprocessing
#=====================================
# get relevant columns only
train_trans = train.copy().drop(columns=['Name' , 'New_Price'])

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

# create transformation pipeline
convert_columns = ['Mileage', 'Engine', 'Power']
num_columns = train_trans.select_dtypes('number').columns.to_list()
cat_columns = [cat_col for cat_col in train_trans.select_dtypes(exclude='number').columns if cat_col not in convert_columns]

num_trans1 = Pipeline([('convertnumeric', ConvertNumeric()), 
                       ('impute', SimpleImputer(strategy='median')),
                       ('scaler', StandardScaler())])

num_trans2 = Pipeline([('impute', SimpleImputer(strategy='median')),
                       ('scaler', StandardScaler())])

cat_trans = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                       ('onehot_encode', OneHotEncoder())])

trans_pipeline = ColumnTransformer(
                  transformers=[('num_trans1', num_trans1, convert_columns), 
                                ('num_trans2', num_trans2, num_columns),
                                ('cat_trans', cat_trans, cat_columns)],
                  remainder='passthrough'
)

trans_pipeline.fit_transform(train_trans)

#=====================================
# Train Model 
#=====================================
X_train_trans = train_trans.drop(['Price'], axis=1)
y_train_trans = train_trans[['Price']]

linear_reg = LinearRegression()
linear_reg.fit(X_train_trans, y_train_trans)

# evaluate model performance on training set
rmse = np.sqrt(mean_squared_error(y_train_trans, linear_reg.predict(X_train_trans)))
print('RMSE: ' + str(rmse))

# evaluate model performance using cross validation
mse_scores = cross_val_score(linear_reg, X_train_trans, y_train_trans, 
                cv=10, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-mse_scores)

print('K-folds RMSE: ', np.mean(rmse_scores), np.std(rmse_scores))

#=====================================
# Automate Preprocessing
#=====================================


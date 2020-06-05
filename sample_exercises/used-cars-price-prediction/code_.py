import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#=====================
# Import Data
#=====================
train = pd.read_csv('data/train-data.csv').iloc[:, 1:]
test = pd.read_csv('data/test-data.csv').iloc[:, 1:]

#=====================
# Data Exploration
#=====================
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

#=====================
# Correlation 
#=====================
train_corr = train.copy()
train_corr.corr()['Price']

#=====================
# Data Preprocessing
#=====================
train_trans = train.copy()
train_trans.loc[:, 'Mileage'] = train_trans.apply(lambda x: x.Mileage.split(' ')[0], axis=1)
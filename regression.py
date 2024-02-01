!pip install lenskit

import pandas as pd
from lenskit.algorithms.bias import Bias
from lenskit.batch import predict
from lenskit.metrics.predict import rmse
import numpy as np

train_features = pd.read_csv('train_features.csv')
train_label = pd.read_csv('train_label.csv')
train_data = pd.merge(train_features, train_label, on='Id')

########## Clean duplicates ##########################
# sorting with timestamp
train_data = train_data.sort_values(by=['user', 'item', 'timestamp'], ascending=[True, True, False])

# # Remove duplicates based on 'user' and 'item', keeping the row with the largest 'timestamp'
train_data = train_data.drop_duplicates(subset=['user', 'item'])

# train_data.describe()
######################################################


########## Clean data with year (2016) ###########
# train_data['timestamp'] = pd.to_datetime(train_data['timestamp'], unit='ms')
startYear = pd.Timestamp("2016-01-01").timestamp()
## Filter rows where the year is 2016 or later
train_data = train_data[train_data['timestamp'] >= startYear]

# Extract the year from the 'timestamp' column
# train_data['timestamp'] = pd.to_datetime(train_data['timestamp'], unit='ms')
# train_data['timestamp'] = train_data['timestamp'].dt.strftime('%Y')
##################################################


######## Train and evaluate ##############
ratings = train_data
validation = ratings.iloc[:1000]
train = ratings.iloc[1000:]
train.head()
algo = Bias()
algo.fit(train)
preds = predict(algo, validation)
print(preds.head())
print(validation.head())
# rmse_value = user_metric(preds, metric=rmse)
rmse_value = rmse(preds['prediction'], preds['rating'], missing='error')
print(f'RMSE: {rmse_value}')
#############################################


test_features = pd.read_csv('test_features.csv')
preds = predict(algo, test_features)
result_df = pd.merge(test_features, preds, on=['user', 'item'])
print(result_df.head())
### write to file
result_df = result_df.rename(columns={'prediction': 'Predicted'})
result_df.head()
result_df.to_csv("predictions.csv", columns=['Id', 'Predicted'], index=False)
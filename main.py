# linear regresion 
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import pandas as pd
# local 

from prepare import prepare_data, train_frame, test_frame
from ploter import plot_milage_vs_price

train_frame, test_frame = prepare_data(train_frame, test_frame)
# lets do some linear regression
# 1. Separate features (X) and target (y)   

def get_model(train_frame):
    X_train = train_frame.drop(columns=['price'])  # Drop the target column 'price'
    y_train = train_frame['price']  # The target variable is 'price'


    # 3. Initialize the linear regression model
    model = LinearRegression()

    # 4. Train the model
    model.fit(X_train, y_train)

    return model

# Function to make predictions
def make_predictions(model, test_frame):
    # Prepare the test data (drop the target column if it exists)
    X_test = test_frame
    predictions = model.predict(X_test)  # Make predictions
    return predictions


if __name__ == '__main__':
    linear_model = get_model(train_frame)
    
    predictions = make_predictions(linear_model, test_frame)

    sample = pd.read_csv('./sample_submission.csv')
    sample['price'] = predictions
    print(sample)
    sample.to_csv('my_submission.csv', index=False)

    
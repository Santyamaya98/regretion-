# linear regresion 
from sklearn.linear_model import LinearRegression


# local 
from prepare import prepare_data, train_frame, test_frame


print(test_frame)
print('********'*4)
print('lets do some magic know')
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
    # Make predictions on the test dataset
    predictions = make_predictions(linear_model, test_frame)

    # Combine the predictions with the test data for better visualization
    results = test_frame.copy()  # Create a copy of the test frame
    results['predicted_price'] = predictions  # Add predictions as a new column

    # Display the results
    print(results[['predicted_price']])  # Print only the predicted prices
    # If you want to see more features alongside the predictions, you can do:
    # print(results)  # Uncomment this to see the entire dataframe with predictions
    

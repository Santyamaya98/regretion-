# linear regresion 
from sklearn.linear_model import LinearRegression


# local 
from prepare import train_frame


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
    # now lets prepare the test.csv 

if __name__ == '__main__':
    
    linear_model = get_model(train_frame)
    print(linear_model)
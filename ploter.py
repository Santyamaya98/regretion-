import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn.linear_model import LinearRegression

# Function to train the model
def get_model(train_frame):
    X_train = train_frame.drop(columns=['price'])  # Separate features
    y_train = train_frame['price']  # Target variable
    model = LinearRegression()
    model.fit(X_train, y_train)  # Fit the model
    return model

# Function to make predictions
def make_predictions(model, test_frame):
    X_test = test_frame  # Prepare test data
    predictions = model.predict(X_test)  # Make predictions
    return predictions

# Function to plot the distribution of predicted prices
def plot_predictions(results):
    sns.histplot(results['predicted_price'], kde=True, color='blue')
    plt.title('Distribution of Predicted Prices', fontsize=16)
    plt.xlabel('Predicted Price', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.savefig('predicted_price_distribution.png', dpi=300)

# Function to plot prices by model year
def plot_price_by_year(train_frame):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='model_year', y='price', data=train_frame)
    plt.title('Price Distribution by Model Year', fontsize=16)
    plt.xlabel('Model Year', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.xticks(rotation=45)
    plt.savefig('plot_price_by_year.png', dpi=300)

# Function to plot the correlation matrix
def plot_correlation_matrix(train_frame):
    corr_matrix = train_frame.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap of Correlation Matrix', fontsize=16)
    plt.savefig('plot_correlation_matrix.png', dpi=300)

# Function to plot feature importance
def plot_feature_importance(model, train_frame):
    if hasattr(model, 'coef_'):
        feature_importance = pd.Series(model.coef_, index=train_frame.drop(columns=['price']).columns)
        feature_importance = feature_importance.sort_values(ascending=False)
        sns.barplot(x=feature_importance.values, y=feature_importance.index)
        plt.title('Feature Importance in Linear Model', fontsize=16)
        plt.xlabel('Coefficient', fontsize=14)
        plt.ylabel('Variable', fontsize=14)
        plt.savefig('plot_feature_importance.png', dpi=300)
    else:
        print("The model does not have the attribute 'coef_'.")

# Function to plot relationships between variables
def plot_pairplot(train_frame):
    sns.pairplot(train_frame[['price', 'model_year', 'milage']])
    plt.savefig('plot_pairplot.png', dpi=300)

# Function to plot mileage vs. price
def plot_milage_vs_price(train_frame):
    plt.scatter(train_frame['milage'], train_frame['price'], color='orange', alpha=0.5)
    plt.title('Relationship between Mileage and Price', fontsize=16)
    plt.xlabel('Mileage', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.savefig('plot_milage_vs_price.png', dpi=300)

# Function to plot accidents vs. price
def plot_accident_vs_price(train_frame):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='accident', y='price', data=train_frame, hue='accident', palette="Set2", dodge=True)
    plt.title('Price Distribution by Accident History', fontsize=16)
    plt.xlabel('Has Had Accidents?', fontsize=14)
    plt.ylabel('Price (in dollars)', fontsize=14)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xticks([0, 1], ['No', 'Yes'], fontsize=12)
    plt.savefig('plot_accident_vs_price.png', dpi=300)



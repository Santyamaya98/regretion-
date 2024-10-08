#data frame
import pandas as pd
import numpy as np
# skelearn
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

#category_encoders
import category_encoders as ce




# load data
test_frame = pd.read_csv(filepath_or_buffer='./test.csv')
train_frame = pd.read_csv(filepath_or_buffer='./train.csv') 

# Look the data
#print(train_frame.info())
#print(train_frame.head())
#print(test_frame.info())
#print(test_frame.head())

# checking for missing values 

#missing_values = train_frame.isnull().sum()
#print(f'missing values for train_set {missing_values}')

#fuel_type        5083
#engine              0
#transmission        0
#ext_col             0
#int_col             0
#accident         2452
#clean_title     21419

# lets handle missing values

#lets assume that none reported are = n/a

#missing_test_values = test_frame.isnull().sum()
#print(f'missing_values for test_set {missing_test_values}')

def prepare_data(train_frame, test_frame):

    # fill train missing data
    train_frame['accident'] = train_frame['accident'].fillna('None reported')

    train_frame['clean_title'] = train_frame['clean_title'].fillna('No')

    train_frame['fuel_type'] = train_frame['fuel_type'].fillna('uknown')

    # fill test missing data
    test_frame['accident'] = test_frame['accident'].fillna('None reported')

    test_frame['clean_title'] = test_frame['clean_title'].fillna('No')

    test_frame['fuel_type'] = test_frame['fuel_type'].fillna('uknown')


    #print(train_frame['accident'])
    #print(test_frame['accident'])
    #print(train_frame.columns)  # Verifica el nombre exacto de las columnas


    # 1. Encode categorical variables using One-Hot Encoding
    train_frame = pd.get_dummies(train_frame, columns=['accident', 'clean_title'], drop_first=True)
    test_frame = pd.get_dummies(test_frame, columns=['accident', 'clean_title'], drop_first=True)
    # 2. Target Encoding for 'model'
    model_encoder = ce.TargetEncoder(cols=['model'])
    train_frame['model_encoded'] = model_encoder.fit_transform(train_frame['model'], train_frame['price'])
    test_frame['model_encoded'] = model_encoder.transform(test_frame['model'])
    # 3. Target Encoding for 'brand'
    brand_encoder = ce.TargetEncoder(cols=['brand'])
    train_frame['brand_encoded'] = brand_encoder.fit_transform(train_frame['brand'], train_frame['price'])
    test_frame['brand_encoded'] = brand_encoder.transform(test_frame['brand'])

    # 4. Ordinal Encoding for 'transmission', 'engine', and 'fuel_type'
    ordinal_encoder = OrdinalEncoder()
    train_frame['transmission_encoded'] = ordinal_encoder.fit_transform(train_frame[['transmission']])
    train_frame['engine_encoded'] = ordinal_encoder.fit_transform(train_frame[['engine']])
    train_frame['fuel_type_encoded'] = ordinal_encoder.fit_transform(train_frame[['fuel_type']])
    test_frame['transmission_encoded'] = ordinal_encoder.fit_transform(test_frame[['transmission']])
    test_frame['engine_encoded'] = ordinal_encoder.fit_transform(test_frame[['engine']])
    test_frame['fuel_type_encoded'] = ordinal_encoder.fit_transform(test_frame[['fuel_type']])

    # 5. Label Encoding for 'ext_col' and 'int_col'
    label_encoder = LabelEncoder()
    train_frame['ext_col_encoded'] = label_encoder.fit_transform(train_frame['ext_col'])
    train_frame['int_col_encoded'] = label_encoder.fit_transform(train_frame['int_col'])
    test_frame['ext_col_encoded'] = label_encoder.fit_transform(test_frame['ext_col'])
    test_frame['int_col_encoded'] = label_encoder.fit_transform(test_frame['int_col'])

    # 6. View the dataset
    #print(train_frame.head())
    #print(test_frame.head())

    # clean columns
    columns_to_drop = ['brand', 'model', 'int_col', 'ext_col', 'fuel_type', 'engine', 'transmission']
    train_frame.drop(columns=columns_to_drop, inplace=True)
    test_frame.drop(columns=columns_to_drop, inplace=True)

    ## Verificar el DataFrame después de eliminar las columnas originales
    #print(train_frame.head())


    # make it more clean
    train_frame.rename(columns={
        'brand_encoded': 'brand', 
        'model_encoded': 'model',
        'int_col_encoded':'int_col', 
        'ext_col_encoded':'ext_col',
        'fuel_type_encoded': 'fuel_type',
        'engine_encoded':'engine',
        'transmission_encoded':'transmission',
        'accident_None reported':'accident',
        'clean_title_Yes':'clean_title' 
        }, inplace=True)

    test_frame.rename(columns={
        'brand_encoded': 'brand', 
        'model_encoded': 'model',
        'int_col_encoded':'int_col', 
        'ext_col_encoded':'ext_col',
        'fuel_type_encoded': 'fuel_type',
        'engine_encoded':'engine',
        'transmission_encoded':'transmission',
        'accident_None reported':'accident',
        'clean_title_Yes':'clean_title' 
        }, inplace=True)    
    
    return train_frame, test_frame





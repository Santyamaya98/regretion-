#data frame
import pandas as pd

# plot
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# skelearn
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
#category_encoders
import category_encoders as ce

# load data
test_frame = pd.read_csv(filepath_or_buffer='./test.csv')
train_frame = pd.read_csv(filepath_or_buffer='./train.csv') 

# Look the data
print(train_frame.info())
print(train_frame.head())
print(test_frame.head())
# Get info for variables in data

print(train_frame['price'].describe())
print('****'*6)
print(test_frame['milage'].describe())

# checking for missing values 

missing_values = train_frame.isnull().sum()
print(f'estos son los valores faltantes {missing_values}')

#fuel_type        5083
#engine              0
#transmission        0
#ext_col             0
#int_col             0
#accident         2452
#clean_title     21419

# lets handle missing values

#lets assume that none reported are = n/a

train_frame['accident'] = train_frame['accident'].fillna('None reported')

train_frame['clean_title'] = train_frame['clean_title'].fillna('No')

train_frame['fuel_type'] = train_frame['fuel_type'].fillna('uknown')

print(train_frame['accident'])
print(train_frame['clean_title'])
print(train_frame.columns)  # Verifica el nombre exacto de las columnas


# 1. Encode categorical variables using One-Hot Encoding
train_frame = pd.get_dummies(train_frame, columns=['accident', 'clean_title'], drop_first=True)

# 2. Target Encoding for 'model'
model_encoder = ce.TargetEncoder(cols=['model'])
train_frame['model_encoded'] = model_encoder.fit_transform(train_frame['model'], train_frame['price'])

# 3. Target Encoding for 'brand'
brand_encoder = ce.TargetEncoder(cols=['brand'])
train_frame['brand_encoded'] = brand_encoder.fit_transform(train_frame['brand'], train_frame['price'])

# 4. Ordinal Encoding for 'transmission', 'engine', and 'fuel_type'
ordinal_encoder = OrdinalEncoder()
train_frame['transmission_encoded'] = ordinal_encoder.fit_transform(train_frame[['transmission']])
train_frame['engine_encoded'] = ordinal_encoder.fit_transform(train_frame[['engine']])
train_frame['fuel_type_encoded'] = ordinal_encoder.fit_transform(train_frame[['fuel_type']])

# 5. Label Encoding for 'ext_col' and 'int_col'
label_encoder = LabelEncoder()
train_frame['ext_col_encoded'] = label_encoder.fit_transform(train_frame['ext_col'])
train_frame['int_col_encoded'] = label_encoder.fit_transform(train_frame['int_col'])

# 6. View the dataset
print(train_frame.head())



#lets look the datset
print(train_frame[['model', 'model_encoded', 'brand', 'brand_encoded']].head())

# Eliminar las columnas originales después de la codificación
print(train_frame.columns)
columns_to_drop = ['brand', 'model', 'int_col', 'ext_col', 'fuel_type', 'engine', 'transmission']
train_frame.drop(columns=columns_to_drop, inplace=True)

# Verificar el DataFrame después de eliminar las columnas originales
print(train_frame.head())


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

print(train_frame.head())


print(train_frame.columns)
print(train_frame.head())
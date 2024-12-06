
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # One-Hot Encoding for categorical variables
    data_encoded = pd.get_dummies(data)

    # Label Encoding for specific columns
    label_encoder = LabelEncoder()
    data['country'] = label_encoder.fit_transform(data['country'])
    data['gender'] = label_encoder.fit_transform(data['gender'])

    # Drop unnecessary columns
    if 'customer_id' in data.columns:
        data.drop(columns=['customer_id'], inplace=True)

    return data_encoded

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
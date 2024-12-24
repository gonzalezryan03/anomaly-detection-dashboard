from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from data_processing import load_data, preprocess_data

def detect_anomalies(data):
    """
    Detects anomalies in the data using a machine learning model.

    Args:
        data: The data to be used for anomaly detection
    """
    # 1. Train-test split
    y = data[' Label'] # Target variable
    X = preprocess_data(data) # Features
    y = y[X.index] # Filter y to match the indices of X
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train a machine learning model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    # 3. Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")


    anomaly_predictions = model.predict(preprocess_data(data.drop(' Label', axis=1)))
    return anomaly_predictions

# Example usage:
folder_path = '/Users/ryangonzalez/anomaly-detection-dashboard/dataset'
data = load_data(folder_path)
anomaly_predictions = detect_anomalies(data.copy())
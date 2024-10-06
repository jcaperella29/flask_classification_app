from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np

def Build_and_use_NN(data, feature_cutoff=0):
    """
    Trains a neural network on the input dataset for classification, selecting the top N features.
    Returns predictions, AUC, and other classification metrics.
    """
    # Separate features (X), labels (y), and sample numbers (first column)
    X = data.iloc[:, 1:-1]  # Features (excluding SampleID and target)
    y = data.iloc[:, -1]    # Target labels
    sample_numbers = data.iloc[:, 0]  # Sample numbers (first column)

    # Convert labels to binary using LabelEncoder for classification
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Convert labels to binary (0, 1)

    # Check if all features are numeric; if not, convert or drop non-numeric columns
    X = X.select_dtypes(include=[np.number])  # Only keep numeric columns

    # Perform feature selection using RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y_encoded)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Select the top N features (based on the feature_cutoff as the number of features to select)
    important_features = feature_importance['Feature'].head(feature_cutoff)

    # Check if any features are selected
    if important_features.empty:
        raise ValueError("No features selected. Please check the feature cutoff value.")

    # Filter the dataset to only include important features
    X_filtered = X[important_features]

    # Standardize the features
    PredictorScaler = StandardScaler()
    X_filtered = PredictorScaler.fit_transform(X_filtered)

    # Split the data into training and testing sets, including sample numbers
    X_train, X_test, y_train, y_test, sample_train, sample_test = train_test_split(
        X_filtered, y_encoded, sample_numbers, test_size=0.3, random_state=42
    )

    # Build and train the neural network for classification
    model = Sequential()
    model.add(Dense(units=5, input_dim=X_filtered.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))  # Sigmoid for binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit(X_train, y_train, batch_size=20, epochs=50, verbose=1)

    # Generate predictions on the test set
    predictions = model.predict(X_test)

    # Convert probabilities to binary for classification
    binary_predictions = (predictions > 0.5).astype(int)

    # Convert binary predictions back to original labels
    predicted_labels = le.inverse_transform(binary_predictions.flatten().astype(int))

    # Keep predictions as probabilities for AUC
    auc = roc_auc_score(y_test, predictions)

    # Compute confusion matrix and classification metrics
    tn, fp, fn, tp = confusion_matrix(y_test, binary_predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    prevalence = (tp + fn) / len(y_test) if len(y_test) > 0 else None

    # Build a proper DataFrame with the index, sample number, and predicted labels
    result_df = pd.DataFrame({
        'Index': range(1, len(sample_test) + 1),  # Adding an index column starting from 1
        'Sample': sample_test.reset_index(drop=True),  # Sample numbers from the test set
        'Predicted': predicted_labels  # Predicted labels
    })

    # Return the DataFrame and metrics
    return result_df, auc, sensitivity, specificity, prevalence

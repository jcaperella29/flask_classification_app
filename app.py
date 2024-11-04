import os
import pandas as pd
from flask import Flask, render_template, request, session, send_file
import umap
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.getcwd(), 'uploads'))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route for home page with file upload, feature selection, and model running options
@app.route("/", methods=["GET", "POST"])
def index():
    predictions_file = None
    metrics_file = None
    predictions = None
    metrics = None
    feature_importance = None
    filepath = session.get('filepath', None)
    max_features = None

    if request.method == "POST":
        print("Index route triggered")

        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            print(f"File uploaded and saved to: {filepath}")
            file.save(filepath)
            session['filepath'] = filepath

        if not filepath:
            return "No file uploaded. Please upload a CSV or Pickle file.", 400

        try:
            if filepath.endswith(".pickle"):
                data = pd.read_pickle(filepath)
            elif filepath.endswith(".csv"):
                data = pd.read_csv(filepath)
            else:
                return "Unsupported file format. Please upload a CSV or Pickle file.", 400

            print(f"Data loaded successfully. Shape: {data.shape}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return f"Error loading data: {str(e)}", 500

        max_features = data.shape[1] - 2
        session['max_features'] = max_features

        X = data.iloc[:, 1:-1]
        y = data.iloc[:, -1]

        # Feature Selection Route
        if 'feature_selection' in request.form:
            try:
                print("Running feature selection...")
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                session['selected_features'] = feature_importance['Feature'].tolist()
                feature_file = os.path.join(UPLOAD_FOLDER, "feature_importance.csv")
                feature_importance.to_csv(feature_file, index=False)
                print("Feature selection completed successfully.")
            except Exception as e:
                print(f"Error during feature selection: {str(e)}")
                return f"Error during feature selection: {str(e)}", 500

            return render_template("index.html", 
                                   feature_importance=feature_importance.to_html(classes='table table-striped'), 
                                   max_features=max_features,
                                   active_tab="feature_importance")

        # Run Model Route
        if 'run_model' in request.form:
            print("Running neural network model...")
            top_features = int(request.form.get("top_x_features", 0))

            # Use selected features or all features if cutoff not set
            selected_features = session.get('selected_features', X.columns)
            if top_features > 0:
                selected_features = selected_features[:top_features]

            X_selected = X[selected_features]
            predictions, metrics = build_and_use_nn(X_selected, y, data.iloc[:, 0])  # Passing sample numbers as well

            return render_template("index.html", 
                                   predictions=predictions.to_html(classes='table table-striped'), 
                                   metrics=metrics,
                                   active_tab="predictions")

    return render_template("index.html", 
                           predictions_file=predictions_file, 
                           metrics_file=metrics_file, 
                           feature_importance=feature_importance, 
                           max_features=max_features)

# Function for building and using the neural network
def build_and_use_nn(X, y, sample_numbers):
    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Standardize features
    PredictorScaler = StandardScaler()
    X_scaled = PredictorScaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test, sample_train, sample_test = train_test_split(
        X_scaled, y_encoded, sample_numbers, test_size=0.3, random_state=42
    )

    # Build neural network
    model = Sequential()
    model.add(Dense(units=5, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit(X_train, y_train, batch_size=20, epochs=50, verbose=1)

    # Predictions and metrics
    predictions = model.predict(X_test)
    binary_predictions = (predictions > 0.5).astype(int)
    predicted_labels = le.inverse_transform(binary_predictions.flatten().astype(int))

    auc = roc_auc_score(y_test, predictions)
    tn, fp, fn, tp = confusion_matrix(y_test, binary_predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    prevalence = (tp + fn) / len(y_test) if len(y_test) > 0 else None

    metrics = {
        'AUC': auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Prevalence': prevalence
    }

    result_df = pd.DataFrame({
        'Index': range(1, len(sample_test) + 1),
        'Sample': sample_test.reset_index(drop=True),
        'Predicted': predicted_labels
    })

    return result_df, metrics

# UMAP Generation Route
@app.route("/generate_umap", methods=["POST"])
def generate_umap():
    print("UMAP route triggered")
    try:
        filepath = session.get('filepath', None)
        if not filepath:
            return "No file found. Please upload a file first.", 400

        data = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_pickle(filepath)
        sample_numbers = data.iloc[:, 0]
        X = data.iloc[:, 1:-1]
        y = data.iloc[:, -1]

        reducer = umap.UMAP()
        embedding = reducer.fit_transform(X)

        umap_fig = px.scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            color=y.astype(str),
            hover_name=sample_numbers,
            hover_data={'Label': y.astype(str)},
            labels={'x': 'UMAP Dim 1', 'y': 'UMAP Dim 2'},
            title="UMAP Visualization"
        )

        umap_html = os.path.join(UPLOAD_FOLDER, 'umap_plot.html')
        umap_fig.write_html(umap_html)

        return send_file(umap_html, as_attachment=True)

    except Exception as e:
        print(f"Error generating UMAP: {str(e)}")
        return f"Error generating UMAP: {str(e)}", 500

# PCA Generation Route
@app.route("/generate_pca", methods=["POST"])
def generate_pca():
    print("PCA route triggered")
    try:
        filepath = session.get('filepath', None)
        if not filepath:
            return "No file found. Please upload a file first.", 400

        data = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_pickle(filepath)
        sample_numbers = data.iloc[:, 0]
        X = data.iloc[:, 1:-1]
        y = data.iloc[:, -1]

        pca = PCA(n_components=2)
        pca_embedding = pca.fit_transform(X)

        pca_fig = px.scatter(
            x=pca_embedding[:, 0],
            y=pca_embedding[:, 1],
            color=y.astype(str),
            hover_name=sample_numbers,
            hover_data={'Label': y.astype(str)},
            labels={'x': 'PCA Dim 1', 'y': 'PCA Dim 2'},
            title="PCA Visualization"
        )

        pca_html = os.path.join(UPLOAD_FOLDER, 'pca_plot.html')
        pca_fig.write_html(pca_html)

        return send_file(pca_html, as_attachment=True)

    except Exception as e:
        print(f"Error generating PCA: {str(e)}")
        return f"Error generating PCA: {str(e)}", 500

@app.route("/download_metrics")
def download_metrics():
    metrics_file = os.path.join(UPLOAD_FOLDER, "metrics.csv")
    if os.path.exists(metrics_file):
        return send_file(metrics_file, as_attachment=True)
    else:
        return "Metrics file not found.", 404

@app.route("/download_predictions")
def download_predictions():
    predictions_file = os.path.join(UPLOAD_FOLDER, "predictions.csv")
    if os.path.exists(predictions_file):
        return send_file(predictions_file, as_attachment=True)
    else:
        return "Predictions file not found.", 404


# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


  

import os
import pandas as pd
from flask import Flask, render_template, request, session, send_file
import umap
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nn_model import Build_and_use_NN

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.getcwd(), 'uploads'))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

        # Check for uploaded file
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            print(f"File uploaded and saved to: {filepath}")
            file.save(filepath)
            session['filepath'] = filepath

        if not filepath:
            return "No file uploaded. Please upload a CSV or Pickle file.", 400

        # Load dataset
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

        max_features = data.shape[1] - 2  # Exclude sample ID and target column
        session['max_features'] = max_features

        X = data.iloc[:, 1:-1]  # Features
        y = data.iloc[:, -1]    # Labels

        # Feature Selection Logic (No cutoff, just rank all features)
        if 'feature_selection' in request.form:
            try:
                print("Running feature selection...")
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                # Save ranked feature importance without any cutoff
                session['selected_features'] = feature_importance['Feature'].tolist()
                feature_file = os.path.abspath(os.path.join(UPLOAD_FOLDER, "feature_importance.csv"))
                feature_importance.to_csv(feature_file, index=False)
                print("Feature selection completed successfully.")
            except Exception as e:
                print(f"Error during feature selection: {str(e)}")
                return f"Error during feature selection: {str(e)}", 500

            return render_template("index.html", 
                                   feature_importance=feature_importance.to_html(classes='table table-striped'), 
                                   max_features=max_features,
                                   active_tab="feature_importance")

        # Model Execution Logic (Cutoff applies here during actual model run)
        if 'run_model' in request.form:
            try:
                top_x_features = int(request.form.get('top_x_features', 10))
                max_features = session.get('max_features', None)
                print(f"Running model with top {top_x_features} features.")

                if top_x_features > max_features:
                    return f"Error: You cannot select more than {max_features} features.", 400

                selected_features = session.get('selected_features', None)
                if selected_features is None:
                    return "Please run feature selection first.", 400

                # Use the selected top X features
                X_selected = X[selected_features[:top_x_features]]
                if X_selected.empty:
                    return "No features selected. Please try running feature selection again.", 400

                print(f"Selected features: {X_selected.columns.tolist()}")

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
                print("Data split for training and testing.")

                # Call Neural Network Model
                result_df, auc, sensitivity, specificity, prevalence = Build_and_use_NN(
                    pd.concat([X_train, y_train], axis=1),
                    feature_cutoff=top_x_features
                )

                print(f"Model execution completed. AUC: {auc}, Sensitivity: {sensitivity}, Specificity: {specificity}")

                # Save predictions and metrics as CSV
                predictions_file = os.path.abspath(os.path.join(UPLOAD_FOLDER, "predictions.csv"))
                result_df.to_csv(predictions_file, index=False)

                metrics = {
                    'AUC': auc,
                    'Sensitivity': sensitivity,
                    'Specificity': specificity,
                    'Prevalence': prevalence
                }

                metrics_file = os.path.abspath(os.path.join(UPLOAD_FOLDER, "metrics.csv"))
                pd.DataFrame.from_dict(metrics, orient='index', columns=['Value']).to_csv(metrics_file)

                # Pass predictions and metrics to the template
                return render_template("index.html", 
                                       predictions=result_df.to_html(classes='table table-striped'),
                                       metrics=metrics,
                                       predictions_file=predictions_file,
                                       metrics_file=metrics_file,
                                       active_tab="predictions")

            except Exception as e:
                print(f"Error during model execution: {str(e)}")
                return f"Error during model execution: {str(e)}", 500

    return render_template("index.html", 
                           predictions_file=predictions_file, 
                           metrics_file=metrics_file, 
                           feature_importance=feature_importance, 
                           max_features=max_features)


# ** UMAP Generation Route **
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

        # Create UMAP embedding
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(X)

        # Create Plotly figure for UMAP
        umap_fig = px.scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            color=y.astype(str),
            hover_name=sample_numbers,
            hover_data={'Label': y.astype(str)},
            labels={'x': 'UMAP Dim 1', 'y': 'UMAP Dim 2'},
            title="UMAP Visualization"
        )

        # Save the UMAP plot as an HTML file
        umap_html = os.path.join(UPLOAD_FOLDER, 'umap_plot.html')
        umap_fig.write_html(umap_html)

        print("UMAP saved as HTML successfully.")

        return send_file(umap_html, as_attachment=True)

    except Exception as e:
        print(f"Error generating UMAP: {str(e)}")
        return f"Error generating UMAP: {str(e)}", 500


# ** PCA Generation Route **
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

        # Create PCA embedding
        pca = PCA(n_components=2)
        pca_embedding = pca.fit_transform(X)

        # Create Plotly figure for PCA
        pca_fig = px.scatter(
            x=pca_embedding[:, 0],
            y=pca_embedding[:, 1],
            color=y.astype(str),
            hover_name=sample_numbers,
            hover_data={'Label': y.astype(str)},
            labels={'x': 'PCA Dim 1', 'y': 'PCA Dim 2'},
            title="PCA Visualization"
        )

        # Save the PCA plot as an HTML file
        pca_html = os.path.join(UPLOAD_FOLDER, 'pca_plot.html')
        pca_fig.write_html(pca_html)

        print("PCA saved as HTML successfully.")

        return send_file(pca_html, as_attachment=True)

    except Exception as e:
        print(f"Error generating PCA: {str(e)}")
        return f"Error generating PCA: {str(e)}", 500


# ** Download Routes for Predictions and Metrics **
@app.route("/download_predictions")
def download_predictions():
    predictions_file = os.path.join(UPLOAD_FOLDER, "predictions.csv")
    if os.path.exists(predictions_file):
        return send_file(predictions_file, as_attachment=True)
    else:
        return "Predictions file not found.", 404

@app.route("/download_metrics")
def download_metrics():
    metrics_file = os.path.join(UPLOAD_FOLDER, "metrics.csv")
    if os.path.exists(metrics_file):
        return send_file(metrics_file, as_attachment=True)
    else:
        return "Metrics file not found.", 404

if __name__ == "__main__":
    app.run(debug=True)

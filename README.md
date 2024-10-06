Readme file for  JCAP Classification, UMAP, and PCA

What the program is and its purpose
This is a Flask web application (app) written in Python, HTML, and CSS. The app uses data in the form of a “.csv” or “.pickle" or ".pkl"  file  and uses that data  to run feature selection, create and evaluate a Keras deep learning model, as well as generate an interactive and downloadable UMAP and PCA  visualization of the data.
Mechanics
Getting started
If running the app locally, you must gather all folders and files from the GitHub repository into one folder. Then in your  command line, type python3  and the path to the app.py file. Please note that  running the app locally requires you to  have Flask, Pandas, NumPy, TensorFlow,  Keras and scikit-learn installed in your virtual environment. The app will be found at http://127.0.0.1:5000. 
Using the app
Function 1-  Feature Selection
1.To begin, look to left of the interface, and you will see a sidebar titled “Controls.”  Beneath that label you will see a “Choose File”   button with “Upload CSV or Pickle” written right above it. This will allow you to upload your data. The last bits of the path to the file containing your data will be in the space next to the “Choose File” label when it is uploaded.
2. Next, click the gray button labeled “Run Feature Selection.” This will cause the app to use a Random Forest classifier to rank features (predictor variables) in your dataset with regard to how much they help classify the data in terms of information score. The ranking will appear as a table in the “Feature Importance” tab. 
Function 2- Deep Learning Classification with Selected Features
1.Beneath the “Run Feature Selection” button you will see a text box that has a red label above it that states ”Top Features.” The value you put in there  will be the number of features that will be used to build the deep learning classifier.
2.Beneath that text box,  you will find a  blue button that is labeled ”Run Model.” This button will cause the app  to train and evaluate a Keras Deep Learning classification  model  on your data. The predictions and the performance metrics (AUC, sensitivity, specificity, and prevalence) will be shown on respective tables in the “Model Predictions” and “Metrics” tabs. When the tables have been generated,  green buttons will  appear beneath the tables allowing you to download the tables as .csv files. 
Functions 3 and 4- Interactive UMAP and PCA plots
Beneath the “Run Model” button you will see two last buttons on the sidebar labeled ”Download UMAP” and “Download PCA.” This button causes a UMAP plot and PCA plot of your data to be generated and downloaded as HTML documents. The plots are colored based on class (a legend is provided ) and hovering over a point provides the sample number associated with that point. Within each plot there  is a camera button allowing you to download the respective plot as a .png file.

Note: After clicking the “Run Feature Selection” or “Run Model” buttons, data will need to be re-uploaded. This is to save memory.

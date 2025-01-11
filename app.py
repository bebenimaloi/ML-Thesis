from flask import Flask, render_template, request
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file uploads and ML model processing
@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    if file:
        # Save the uploaded file to a temporary directory
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Read the CSV file
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            return f"Error reading the file: {e}", 400

        # Check for required columns in the dataset
        required_columns = [
            'Cognitive Abilities', 'Engagement Levels', 'Pre-Test Scores',
            'Post-Test Scores', 'Time Spent on AR', 'Frequency of AR Use', 'Performance'
        ]
        if not all(col in data.columns for col in required_columns):
            return f"The dataset must contain the following columns: {', '.join(required_columns)}", 400

        # Prepare feature columns and target
        X = data[['Cognitive Abilities', 'Engagement Levels', 'Pre-Test Scores',
                  'Post-Test Scores', 'Time Spent on AR', 'Frequency of AR Use']]
        y = data['Performance']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Choose the ML model
        model_choice = request.form.get('model')
        if model_choice == 'random_forest':
            model = RandomForestClassifier(random_state=42)
        elif model_choice == 'decision_tree':
            model = DecisionTreeClassifier(random_state=42)
        else:
            return "Invalid model choice", 400

        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Create a visualization
        plt.figure(figsize=(10, 6))
        feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        if feature_importances is not None:
            plt.barh(X.columns, feature_importances)
            plt.xlabel("Feature Importance")
            plt.ylabel("Feature")
            plt.title("Feature Importance")
            plt.tight_layout()
            plot_path = os.path.join(upload_folder, "feature_importance.png")
            plt.savefig(plot_path)
            plt.close()
        else:
            plot_path = None

        # Clean up: Delete the uploaded file
        os.remove(file_path)

        # Render the results template
        return render_template(
            'results.html',
            model_choice=model_choice,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            plot_path=plot_path
        )

# Route for testing
@app.route('/test')
def test():
    print("Test route accessed!")  # This will print to your terminal
    return "Flask is working!"

if __name__ == '__main__':
    app.run(debug=True)

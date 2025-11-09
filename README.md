Disease Prediction from Medical Data – CodeAlpha Task 4

📌 Project Overview

This project implements a comprehensive disease prediction system using Python and Machine Learning. It uses multiple publicly available medical datasets to predict the risk of Heart Disease, Diabetes, and Breast Cancer for patients.

The system trains multiple models, evaluates their performance, and provides ensemble predictions for higher accuracy.

⸻

🚀 Features
	•	Data Loading & Preprocessing: Automatically loads Heart Disease, Diabetes, and Breast Cancer datasets from UCI ML Repository.
	•	Exploratory Data Analysis (EDA): Visualizes target distributions and provides insights on each dataset.
	•	Machine Learning Models: Uses Logistic Regression, Naive Bayes, SVM, KNN, Decision Tree, and Random Forest.
	•	Ensemble Prediction: Combines multiple models for robust predictions.
	•	Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrices.
	•	Patient Prediction Function: Predicts health status for a given patient across all three diseases.
	•	Demonstration Examples: Includes sample patients to illustrate predictions.

⸻

🛠️ Technologies Used
	•	Python 3.x
	•	Libraries:
	•	numpy
	•	pandas
	•	matplotlib
	•	seaborn
	•	scipy
	•	scikit-learn

⸻

📂 Project Structure :
disease_prediction.py   # Main script with full workflow
README.md               # Project documentation

📊 Workflow Steps
	1.	Data Loading & Preprocessing
	•	Loads datasets for Heart Disease, Diabetes, and Breast Cancer.
	•	Handles missing values and converts categorical variables.
	2.	Exploratory Data Analysis
	•	Displays first few records and plots target distributions.
	3.	Model Training
	•	Trains multiple ML models on each dataset.
	•	Evaluates individual models and ensemble performance.
	4.	Prediction Function
	•	predict_health_status(patient_data) predicts all three diseases for a patient.
	•	Handles missing features by using median values from training data.
	5.	Demonstration
	•	Example patient predictions for healthy and high-risk cases.

⸻

🔧 Installation & Usage
	1.	Clone the repository or download the script.
	2.	Install dependencies: pip install numpy pandas matplotlib seaborn scipy scikit-learn
 	3.	Run the script: python disease_prediction.py
  4.	View visualizations & predictions directly in your console and plots window.

⸻

📈 Example Outputs
	•	Distribution plots of disease cases.
	•	Confusion matrices for each model and ensemble.
	•	Performance metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC).
	•	Predictions for example patients including risk probabilities.

⸻

🤝 Contributing

Contributions are welcome! You can enhance the project by:
	•	Adding more real-world datasets.
	•	Testing additional ML models.
	•	Hyperparameter tuning for better predictions.

⸻

📜 License

This project is created as part of CodeAlpha Task 4 and is free to use for educational purposes.

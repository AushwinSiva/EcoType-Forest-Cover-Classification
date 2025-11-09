🌲 Forest Cover Type Prediction — End-to-End ML & Streamlit App

🧠 Overview

This project predicts forest cover types using a combination of environmental, terrain, and soil features from the UCI Forest Cover dataset.
The goal is to build a complete machine learning workflow — from data cleaning and feature engineering to model tuning and deployment through a Streamlit web app.

The dataset contains various physical characteristics such as elevation, slope, distance to hydrology, soil type, and more. Using these features, the model classifies forest areas into one of seven cover types like Spruce/Fir, Lodgepole Pine, Aspen, and Douglas-fir.

🎯 Objectives

* Understand the data through EDA and visualize trends.
* Handle class imbalance and improve feature quality.
* Build and compare multiple ML models.
* Tune the best-performing model for optimal results.
* Save and deploy the final model using Streamlit for real-time prediction.

⚙️ Project Workflow

1️⃣ Data Preparation

1. Loaded and explored the cover_type.csv dataset.
2. Checked data types, missing values, and label distribution.
3. Encoded categorical target labels using LabelEncoder.
4. Split the data into training (80%) and testing (20%) sets.

2️⃣ Exploratory Data Analysis (EDA)

1. Visualized feature distributions using histograms and boxplots.
2. Checked correlations between numeric features using heatmaps.
3. Observed mild class imbalance — certain cover types dominate the dataset.
4. Identified important features such as Elevation, Horizontal_Distance_To_Roadways, and Hillshade_9am.

📊 Key Insight:
Higher elevation areas tend to be dominated by Spruce/Fir and Lodgepole Pine, while lower elevations support Ponderosa Pine and Douglas-fir.

3️⃣ Feature Engineering

* Created new, meaningful features to help the model learn better patterns:
* dist_to_hydrology_euclid = √(Horizontal² + Vertical²)
* road_fire_sum = Distance to Roadways + Distance to Fire Points
* aspect_sin and aspect_cos = Circular transformation of Aspect
* elev_slope = Interaction between elevation and slope
* Dropped low-variance and highly correlated features to simplify training.

4️⃣ Model Building and Comparison

Trained and compared five classification models:

* Model	Accuracy	Macro-F1	Remarks
* Random Forest	0.956	0.91	Best performing
* Decision Tree	0.940	0.88	Slightly less stable
* Logistic Regression	0.484	0.38	Underfits
* KNN	0.945	0.87	Slow for large data
* XGBoost	0.928	0.90	Strong but slower

📌 Conclusion: Random Forest performed the best overall with strong generalization and interpretability.

5️⃣ Hyperparameter Tuning

Used RandomizedSearchCV for fine-tuning the Random Forest on a 100K sample (for speed).
Best parameters obtained:

{
  'n_estimators': 200,
  'max_depth': None,
  'min_samples_split': 5,
  'min_samples_leaf': 2,
  'max_features': 0.5,
  'bootstrap': True
}

Final retraining on the full dataset gave:

✅ Accuracy: 0.9645
✅ Macro F1-score: 0.924

6️⃣ Model Evaluation

* Generated evaluation reports and visualizations:
* Confusion Matrix: highlights strong predictions for dominant forest types.
* Feature Importances: Elevation, Road-Fire Distance, and Hillshade were top drivers.
* Classification Report: saved in final_classification_report.csv.

📊 The model generalizes well with balanced precision and recall across all classes.

7️⃣ Streamlit Web Application

* A clean and interactive Streamlit app allows users to:
* Manually input terrain and environmental parameters.
* Automatically fill missing or less-important features with mean/default values.
* Instantly predict the forest cover type.
* Display the decoded forest name (e.g., “Spruce/Fir”) instead of numeric labels.

💡 Only essential features are user-entered for simplicity — the rest are auto-filled internally for a smoother experience.

🧩 Technologies Used

* Python 3.12
* Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib, xgboost, streamlit, joblib, imbalanced-learn
* Tools: Google Colab, Jupyter Notebook, GitHub, VS Code

📈 Results Summary

* Best Model: Random Forest Classifier
* Accuracy: 96.45%
* Macro F1-Score: 0.924
* Top Features: Elevation, Road-Fire Distance, Hillshade_9am, Aspect
* Deployment: Streamlit-based interactive web app
* The final model shows excellent performance, interpretability, and real-world usability.

🏁 Conclusion

This end-to-end project demonstrates the complete ML lifecycle — from raw data to an interactive deployment-ready model.
It blends data science, machine learning, and software engineering principles into a clear, reproducible pipeline.
The final application empowers users to explore how environmental features influence forest types in a visually engaging way.

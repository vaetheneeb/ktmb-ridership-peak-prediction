# KTMB Peak Hour Classification
This project applies supervised machine learning techniques to classify KTMB Komuter train ridership data as **Peak** or **Non-Peak**. It was completed as part of the Machine Learning course (WIA1006) at Universiti Malaya. 

# Objective
To develop a predictive model that determines whether a train ride occurred during peak hours based on temporal and location-based features. The goal is to support data-driven decisions in transport scheduling and passenger flow optimization. From a syllabus perspective, the objective was to implement and develop at least five models from the various machine learning methods we studied throughout the semester, and to compare their effectiveness as well as understand the reasoning behind their performance when applied to the same problem.

# Tools & Technologies
- **Python** (pandas, scikit-learn, matplotlib)
- **Google Colab**
- **Auto-sklearn** (for automated model comparison)
- **WSL Ubuntu via VSCode**
- **Canva** (final presentation slides)

# Dataset
The data was obtained from [data.gov.my](https://data.gov.my), specifically the **Hourly Origin-Destination Ridership** dataset for KTMB Komuter trains. After through data cleaning & preprocessing, features used are:
- hour
- day_of_week
- is_holiday
- hour_bin
- is_weekend
- hour_is_weekend
- hour_is_holiday
- station_pair_encoded
- origin_encoded
- destination_encoded

# Models Trained
Five supervised learning models were trained and evaluated:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Linear Support Vector Classifier (Linear SVC)
- Random Forest Classifier

Additionally, **Auto-sklearn** and **Autogluon** was used to compare results with automated ML pipelines.

# My Role
• Project Management: Oversaw task division and progress tracking to ensure an organized and timely workflow within the team.

• Model Development: Trained and evaluated a Logistic Regression model, carefully selecting encoding strategies to strike a balance between performance and interpretability

• Model Benchmarking: Integrated and configured Auto-sklearn to compare the performance of five manually tuned models against automated pipelines.

• Presentation & Visual Design: Designed the final presentation slides with a focus on clarity, structure, and effective storytelling to communicate our findings to both technical and non-technical audiences. I opted for a clean, dual-color palette with minimal graphics to highlight the data and results—keeping in mind the 5-minute presentation limit and academic context of the assignment.

# Feature Engineering
To improve model interpretability and performance:
- `station_pair` was encoded using **target encoding** to preserve contextual relationships while reducing dimensionality.
- `day_of_week` and `hour_bin` were encoded using **one-hot encoding** to highlight time-based patterns without imposing order.

# Evaluation
The Logistic Regression model achieved approximately **76% accuracy**, balancing performance and interpretability. Classification reports and model comparisons were used to analyze the strengths and limitations of each approach. The model demonstrated strong generalization, particularly on the dominant non-peak class while maintaining reasonable detection of peak periods.

# AutoML Benchmarking with Auto-sklearn
To evaluate the performance of our manually tuned models, we used Auto-sklearn, an automated machine learning (AutoML) framework that automatically builds model ensembles and optimizes hyperparameters.
🔧 Configuration
Framework: Auto-sklearn
Metric: Accuracy
Total algorithm runs: 12
Successful runs: 9
Timeouts: 3
Crashes: 0
Best validation accuracy: 0.84

🏆 Top Models in the Final Ensemble
Auto-sklearn selected an ensemble **composed entirely of Random Forest classifiers** with varying configurations. The table below shows the top-ranked models based on their validation accuracy and their contribution to the ensemble:

Rank	Validation Accuracy	Ensemble Weight	Notable Hyperparameters
1	~0.82	0.04	max_features=3
2	~0.79	0.06	max_features=7, min_samples_leaf=2, min_samples_split=20
3	~0.80	0.04	criterion='entropy', max_features=1, min_samples_leaf=5
4	~0.83	0.08	bootstrap=False, max_features=10, min_samples_leaf=6
5	~0.82	0.14	criterion='entropy', max_features=2, class_weight=balanced
6	0.84 (Best Overall)	0.50	criterion='entropy', max_features=8, min_samples_leaf=7

📌 Insights
Auto-sklearn consistently **favored Random Forests**, likely due to their robustness on tabular data and ability to handle mixed feature types and imbalanced classes.
The **best individual model achieved 0.84 accuracy**, **similar** to the performance of our manually trained Random Forest model, which had an accuracy of **84.79%**.
Ensemble weighting suggests that multiple diverse configurations contributed meaningfully to the final prediction, with one high-performing model carrying most of the weight.

# Presentation
Final results were compiled into a structured presentation, highlighting:
- The full ML pipeline
- Insights from the data
- Model comparison
- Implications for real-world scheduling and service planning

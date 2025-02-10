# Titanic Survival Prediction with Logistic Regression

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Files](#project-files)
- [Methodology](#methodology)
  - [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Feature Importance Analysis](#feature-importance-analysis)
  - [Predictions on New Unseen Data](#predictions-on-new-unseen-data)
- [Results and Insights](#results-and-insights)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Technologies Used](#technologies-used)
- [How to Run the Project Locally](#how-to-run-the-project-locally)
- [Google Colab Notebook](#google-colab-notebook)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

## Overview
This project aims to predict whether a passenger survived the Titanic disaster using logistic regression. Leveraging the famous [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data) dataset from Kaggle, the project follows a structured machine learning pipeline that includes:
- Data exploration and preprocessing
- Feature engineering
- Model training using logistic regression
- Model evaluation and feature importance analysis
- Predictions for new unseen passenger data

## Dataset
The project uses the **Titanic: Machine Learning from Disaster** dataset, which contains detailed information about the passengers such as:
- **Demographic Information:** Age, Sex
- **Socioeconomic Indicators:** Pclass (ticket class), Fare
- **Family Relationships:** SibSp (number of siblings/spouses aboard), Parch (number of parents/children aboard)
- **Additional Details:** PassengerId, Name, Ticket, Cabin, Embarked

Handling of missing values includes:
- **Age:** Missing values are imputed using the median.
- **Embarked:** Missing values are filled with the most frequent category.
- **Cabin:** Dropped due to a high percentage of missing values.

## Project Files
This repository contains the following files:
- `Titanic_Logistic_Regression.ipynb`: Jupyter Notebook containing code for data preprocessing, model training, and evaluation.
- `2_Logistic_Regression_report.pdf`: A report summarizing the project's findings, including model performance and feature analysis.
- `02. Logistic Regression.pdf`: Instructions outlining the tasks and methodology followed in this project.

## Methodology

### Data Exploration and Preprocessing
- **Data Loading:** Imported and examined the dataset to understand its structure.
- **Missing Values:** Addressed missing values in features such as Age, Embarked, and Cabin.
- **Data Cleaning:** Removed irrelevant features and handled outliers.
- **Visualization:** Explored distributions and relationships between key features.

### Feature Engineering
- **Feature Selection:** Chose important predictors like Pclass, Sex, Age, SibSp, Parch, Fare, and Embarked.
- **Encoding:** Applied one-hot encoding to categorical variables (Sex, Embarked).
- **Normalization:** Scaled numerical features to enhance model performance.

### Model Training
- **Data Splitting:** Divided the dataset into training and testing sets.
- **Logistic Regression:** Implemented logistic regression using both scikit-learn and a custom implementation developed during coursework.

### Model Evaluation
- **Performance Metrics:** Evaluated the model using:
  - **Accuracy:** 80%
  - **Precision:** 87%
  - **Recall:** 87%
  - **F1 Score:** 87%
- **Visual Analysis:** Generated a confusion matrix and ROC curve (AUC = 0.93) to assess predictive performance.

### Feature Importance Analysis
- **Coefficient Analysis:** Examined the logistic regression coefficients to understand the influence of each feature:
  - **Positive Impact:** Being female (Sex), higher SibSp, and Fare increased the likelihood of survival.
  - **Negative Impact:** Higher age, lower Pclass, and certain Embarked categories were associated with lower survival rates.

### Predictions on New Unseen Data
- **Application:** Demonstrated the model's ability to predict survival for new passenger data, showcasing its potential for real-world applications.

## Results and Insights
- **Discriminatory Power:** The ROC curve indicates an AUC of 0.93, demonstrating excellent capability in distinguishing between survivors and non-survivors.
- **Key Predictors:** Gender, fare, and ticket class emerged as significant indicators of survival.
- **Model Performance:** The model's accuracy and balanced precision/recall suggest it is robust for predictive analysis in this context.

## Limitations and Future Improvements
- **Dataset Bias:** The model is affected by the inherent biases present in the dataset.
- **Model Complexity:** While logistic regression is effective, it may not capture complex nonlinear relationships. Future work could incorporate advanced methods such as decision trees or ensemble techniques.
- **Data Enrichment:** Including additional features or real-time data might further enhance predictive performance.

## Technologies Used
- **Python**
- **Pandas & NumPy:** Data processing and manipulation
- **Scikit-learn:** Machine learning implementation
- **Matplotlib & Seaborn:** Data visualization
- **Jupyter Notebook / Google Colab:** Interactive coding and experimentation

## How to Run the Project Locally
To run the project on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/titanic-logistic-regression.git
   cd titanic-logistic-regression
   ```
2. **Install the Required Libraries:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. **Launch Jupyter Notebook:**
    Run the following command to open the Jupyter Notebook in your browser:
    ```bash
    jupyter notebook
    ```
    Open the `2_Logistic_Regression.ipynb` notebook and run the cells to execute the project.

## Google Colab Notebook
Alternatively, you can run the project in the cloud using Google Colab:
[Google Colab Notebook](https://colab.research.google.com/drive/1E9y7q07qpoDp-8bfGEvrZWiJ6O0t7-H2?usp=sharing)

## Contributors
- **Douadjia Abdelkarim**  
  Master 1 Artificial Intelligence, Djilali Bounaama University of Khemis Miliana

## Acknowledgments
- **Kaggle:** For providing the dataset.
- **Scikit-learn:** For the machine learning tools.
- **Djilali Bounaama University:** For academic support.
- **Course Instructors:** For guidance on model development and evaluation.

---
This project is part of coursework on **Machine Learning with Logistic Regression** and aims to provide hands-on experience in predictive modeling.

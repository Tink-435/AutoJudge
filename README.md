# AutoJudge  
## Predicting Programming Problem Difficulty Using Machine Learning

---

## Project Overview

Determining the difficulty of programming problems is an important but often subjective task on online coding platforms. Difficulty is typically represented both as a numerical score and as a categorical label such as *Easy*, *Medium*, or *Hard*.

**AutoJudge** is a machine learning–based system that predicts the difficulty of programming problems using only their textual descriptions. Given a problem statement, the system:

- Predicts a numerical difficulty score using regression  
- Derives a difficulty class (*Easy / Medium / Hard*) from the predicted score  

The project demonstrates how natural language processing (NLP) and machine learning techniques can be applied to analyze and model problem difficulty in a consistent and interpretable way.

---

##  Dataset Used

The dataset is provided in JSON format as `problems_data.json` and contains programming problems collected from online judges.

Each problem includes:
- Problem title  
- Problem description  
- Input description  
- Output description  
- Difficulty class (*Easy / Medium / Hard*)  
- Numerical difficulty score  

The dataset contains **4112 programming problems** and is used directly without manual relabeling.

---

##  Approach and Models Used

### Text Preprocessing
- All relevant text fields (title, description, input, output) are combined  
- Text is converted to lowercase and cleaned  
- This ensures the complete context of each problem is captured  

### Feature Extraction
The following features are used:
- **TF-IDF features** (unigrams and bigrams)  
- **Structural text features**
  - Number of characters  
  - Number of words  
  - Count of mathematical symbols  
- **Keyword-based features**
  - Presence of common algorithmic terms such as *DP, graph, tree, BFS, DFS*  

All features are combined into a single feature vector.

---

##  Models

### Difficulty Score Prediction (Regression)

Multiple regression models were evaluated:
- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  

Among these, the **Random Forest Regressor** achieved the lowest MAE and RMSE values and is therefore used as the final backbone model of the system.

---

### Difficulty Class Prediction

Two strategies were explored:
- Direct multiclass classification (baseline experiment)  
- Regression-based classification, where difficulty classes are derived from predicted difficulty scores  

Although direct classification achieved slightly higher accuracy, it produced inconsistent outputs when compared with predicted difficulty scores. To ensure consistency and interpretability, the final system derives difficulty classes from the predicted score using **fixed thresholds**.

---

##  Evaluation Metrics

### Classification
- Accuracy  

### Regression
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  

Both fixed-threshold and quantile-based classification accuracies are reported in the notebook for comparison and analysis.

---

##  Web Interface

A simple and interactive web interface is built using **Flask**.

### Features
**Input fields for:**
- Problem title  
- Problem description  
- Input description  
- Output description  

**Displays:**
- Predicted difficulty score  
- Derived difficulty class  
- A *“How does this work?”* section explaining the modeling approach  

The application runs locally and loads pre-trained models.

---

## Steps to Run the Project Locally
- Clone the Repository
- Install Dependencies (packages listed in requirements.txt)
- (Optional) Run the Notebook (AutoJudge.ipynb)
- Run the Web Application (run app.py in terminal and then open your      browser and navigate to: http://127.0.0.1:5000/)

---

## Saved Trained Models
The following trained models and preprocessing objects are included in the models/ directory:
- vectorizer.pkl – TF-IDF vectorizer
- scaler.pkl – Feature scaler
- regressor.pkl – Final regression model
- classifier_baseline.pkl – Baseline classification model saved for
  experimental comparison (not used in the final system)

These files are loaded directly by the web application.

---

## Demo Video 
**Duration:** 2 minutes 13 seconds  
**Demo Video Link:** https://drive.google.com/file/d/1QcawggzZ8T2SFvwJh3ZwtLzLwaTMplqJ/view?usp=share_link

The demo video shows:
- Brief explanation of the project
- Model approach
- Working web interface with predictions

---

##  Author

**Name:** Eshika Suresh Katekhaye  
**Enrollment No.:** 24114036  
**Branch:** Computer Science and Engineering (2nd Year)  
**Institution:** Indian Institute of Technology Roorkee  
**Contact:** eshikakatekhaye@gmail.com

---

## Final Notes
- The project runs locally without errors
- All results correspond directly to the submitted code
- No external APIs or hosting services are required
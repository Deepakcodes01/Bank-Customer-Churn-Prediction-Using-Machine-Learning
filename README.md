# Bank Customer Churn Prediction Model 🌟

Welcome to the **Bank Customer Churn Prediction Model** repository! 🚀 This project demonstrates how machine learning techniques can be applied to predict customer churn, a critical metric for banks aiming to retain valuable customers. The project was developed to highlight the power of data preprocessing, feature engineering, and classification models in tackling real-world business challenges.

---

## 🨠 About the Project

### **What is Customer Churn?**

Customer churn occurs when customers stop using a business's products or services. For banking institutions, predicting churn is vital to designing strategies for retaining high-value customers.

### **Objective of the Model**

This project uses machine learning to predict whether a customer is likely to churn based on their profile and account behavior. The solution includes:

- **Data preprocessing** for cleaning and transforming customer data.
- **Feature engineering** to derive meaningful insights.
- **Model training and evaluation** to ensure robust predictions.

---

## 💻 Key Features

### **Data Preprocessing**

- Cleaned and encoded customer data (e.g., gender, geography).
- Addressed class imbalance using techniques like Random Under Sampling (RUS) and Random Over Sampling (ROS).

### **Feature Engineering**

- Added derived features like "Zero Balance."
- Standardized numerical attributes like Credit Score and Balance.

### **Model Training**

- Tested multiple models, selecting the best-performing one (e.g., Logistic Regression, Random Forest).
- Tackled data imbalance and optimized hyperparameters.

### **Prediction and Evaluation**

- Metrics include **confusion matrix**, **precision**, **recall**, and **F1-score**.
- Compared performance across models for robustness.

---

## 📊 How It Works

### **Input Data**

Key customer attributes, such as:

- Credit Score
- Age
- Geography
- Gender
- Number of Products
- Account Balance
- Is Active Member

### **Processing Pipeline**

1. Data Cleaning ➔ Encoding ➔ Resampling ➔ Scaling ➔ Model Training.
2. Output: **Binary classification** (1 = churn, 0 = no churn).

---

## 📁 Repository Structure

```
Bank-Customer-Churn-Prediction/
├── data/
│   ├── Bank_Customer_Churn_Prediction.csv   # Dataset file
├── images/
│   ├── confusion_matrix.png                # Confusion Matrix visualization
│   ├── feature_importance.png              # Feature importance plot
├── notebooks/
│   ├── Bank_Customer_Churn_Model.ipynb     # Jupyter Notebook
├── src/
│   ├── data_preprocessing.py               # Data preprocessing scripts
│   ├── model_training.py                   # Model training scripts
│   ├── model_evaluation.py                 # Model evaluation scripts
│   ├── churn_prediction.py                 # Churn prediction function
├── saved_models/                           # Directory for saved models
├── README.md                               # This README file
├── requirements.txt                        # Python dependencies
```

---

## 🟨 Sample Outputs

### **Output 1: Confusion Matrix**
```plaintext
[[150  30]
 [ 25  95]]
```

### **Output 2: Logistic Regression Classification Report**
```plaintext
              precision    recall  f1-score   support

           0       0.86      0.83      0.84       180
           1       0.76      0.79      0.78       120

    accuracy                           0.82       300
   macro avg       0.81      0.81      0.81       300
weighted avg       0.82      0.82      0.82       300
```

### **Output 3: ROC Curve AUC Scores**
```plaintext
Logistic Regression AUC: 0.87
Random Forest AUC: 0.91
```

---

## 🛠️ How to Use

### **Clone the Repository**

```bash
git clone https://github.com/Deepakcodes01/Bank-Customer-Churn-Prediction-Using-Machine-Learning
cd bank-customer-churn-prediction
```

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Run the Script**

```bash
python src/bank_customer_churn_model.py
```

### **Upload Dataset**

Ensure the dataset is available in the `data/` folder or update the file path in the script.

### **View Outputs**

Check the predictions and evaluation metrics in the console or generated visualizations.

---

## ✨ Technologies Used

- **Programming Language:** Python 🐍
- **Libraries:**
  - Pandas, NumPy
  - Scikit-learn, Imbalanced-learn
  - Matplotlib, Seaborn
- **Cloud Integration:** Ready for AWS, GCP, or Azure for deployment.

---

## 🎯 Future Improvements

- Deploy the model using **Docker** and **Kubernetes**.
- Develop a **REST API** using Flask or FastAPI for real-time predictions.
- Integrate with a cloud storage bucket for dynamic data handling.

---

## 📢 Acknowledgments

- **Mentors and Peers:** For guidance and feedback.
- **Open-source Contributors:** For tools and datasets.

Feel free to fork, star ⭐, and contribute to this project! Let’s make predictive analytics more accessible and impactful. 😊


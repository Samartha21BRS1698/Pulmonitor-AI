# 🫁 Pulmonitor AI

> 💡 An AI-powered ML model cum website which analyses and detects the presence of Pneumonia and other pulmonary diseases based on the provided chest X-ray image and recommends nearby health centres for further treatment

---

## 🚀 Overview

**Pulmonitor AI** is a Machine Learning + Web-based application that:
-  Classifies chest X-ray images into healthy or disease categories (e.g., Pneumonia, Effusion)
-  Suggests nearest hospitals or health centers for follow-up
-  Built with Jupyter + Transfer Learning (ResNet50) + Web frontend (Streamlit)

---

## 📦 Features

- Deep Learning-based X-ray image classification
- Pre-trained ResNet50 model (Transfer Learning)
- Confusion Matrix, Accuracy, and AUC evaluation
- Web interface for image upload and prediction
- Geo-based hospital suggestion (optional with APIs)

---

## 🧠 Technologies Used

| Component       | Tools / Libraries                         |
|-----------------|-------------------------------------------|
| Model           | Python, Keras, TensorFlow, ResNet50       |
| Image Processing| OpenCV, NumPy, Matplotlib                 |
| Web App         | Flask or Streamlit (if used)              |
| Jupyter Notebook| Training + EDA workflow                   |
| Deployment      | GitHub, Localhost, Streamlit              |

---
---
## 🧾 Dataset

- 📁 Source: [Kaggle Chest X-ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Classes: `Normal`, `Pneumonia`

---

## 🧰 Installation

```bash
# Clone the repository
git clone https://github.com/Samartha21BRS1698/Pulmonitor-AI.git
cd Pulmonitor-AI

# Create environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## 🧪 Results & Evaluation

| Metric           | Value         |
|------------------|---------------|
| Accuracy         | 93.4%         |
| AUC Score        | 0.95          |
| Confusion Matrix | Provided      |
| Model Used       | ResNet50      |

*Training + testing using Kaggle Chest X-ray dataset.*

System architecture: 
![image](https://github.com/user-attachments/assets/10a7b278-c37c-4eeb-ae7c-23681d6ea71f)


📁 Folder Structure

```bash
Pulmonitor-AI/
│
├── notebooks/
│   └── Pulm3_Transfer.ipynb     # Training notebook
├── Pulmonitor_UI_Website/       # Web frontend (if built)
├── models/                      # Trained model files (.h5)
├── static/                      # Web assets
├── templates/                   # HTML files (if Flask used)
├── requirements.txt
├── README.md
└── .gitignore
```

## 🙌 Author

**Samartha**  
B.Tech student 
🎓 AI/ML • Data Science •  NLP • Google Cloud 
🔗 [LinkedIn](https://www.linkedin.com/in/samartha-b0154a293) | [GitHub](https://github.com/Samartha21BRS1698)

📝 License
 MIT License © 2025 Samartha

 
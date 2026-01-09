🦠 Malaria Detection Using Deep Learning

An end-to-end deep learning system that automatically detects malaria parasites from microscopic blood smear images using Convolutional Neural Networks (CNNs), enabling fast, accurate, and scalable malaria diagnosis in clinical and remote healthcare environments. 

Malaria detection using deep le…

📌 Project Overview

Malaria remains a life-threatening disease, especially in rural and resource-constrained regions where expert pathologists are scarce. Traditional microscopic diagnosis is time-consuming and error-prone. This project automates malaria detection using CNN models trained on microscopic cell images to classify them as Parasitized or Uninfected with very high accuracy.

🚀 Features

Automated malaria detection from blood smear images

Custom CNN achieving 98.99% validation accuracy

Transfer learning using MobileNetV2 & EfficientNetB0

Flask-based web application for real-time predictions

Confidence score & probability distribution display

Health-check API endpoint for system monitoring

🛠️ Tech Stack
Layer	Technology
Backend	Python, Flask
DL Framework	TensorFlow, Keras
Models	Custom CNN, MobileNetV2, EfficientNetB0
Image Processing	Pillow, NumPy
Visualization	Matplotlib, Seaborn
Dataset	NIH Malaria Dataset (Kaggle)
📂 Project Structure
<img width="746" height="930" alt="image" src="https://github.com/user-attachments/assets/116e55b4-6258-489a-9d59-f0abd37cfc84" />



⚙️ Setup Instructions
git clone https://github.com/your-username/malaria-detection-dl.git
cd malaria-detection-dl
pip install -r requirements.txt
python app1.py


Open browser:

http://127.0.0.1:5000

🎯 Learning Outcomes

CNN & Transfer Learning for medical imaging

Model evaluation with ROC-AUC, confusion matrix

Flask deployment of DL models

Real-time AI inference systems

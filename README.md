# Transfer Learning for an Automated Detection System for Patients with Maxillofacial Trauma (MFDS)

Medical imaging system for **detecting maxillofacial fractures from CT scans** using **Transfer Learning and Convolutional Neural Networks (CNNs).**

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Keras](https://img.shields.io/badge/Keras-NeuralNetworks-red)
![OpenCV](https://img.shields.io/badge/OpenCV-ImageProcessing-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📌 Overview

Maxillofacial trauma is a serious medical condition that requires **fast and accurate diagnosis**. Traditional diagnosis relies on **radiologists manually analyzing CT scans**, which can be time-consuming and prone to human error in high-pressure environments.

This project introduces an **Automated Maxillofacial Fracture Detection System (MFDS)** that uses **deep learning and transfer learning** to automatically classify CT scan images into:

- **Fracture**
- **No Fracture**

The system assists healthcare professionals by **providing decision support during trauma diagnosis**.

Although the system does **not replace radiologists**, it can significantly **reduce diagnostic delays and support clinical decision making**.

---

# 🧠 Abstract

An original **Maxillofacial Fracture Detection System (MFDS)** based on **Convolutional Neural Networks and Transfer Learning** was developed to detect traumatic fractures in patients.

A **pre-trained convolutional neural network**, originally trained on large-scale non-medical image datasets, was **re-trained and fine-tuned using CT scan images** to produce a classification model.

The system predicts whether CT scans belong to:

- **Fracture**
- **No Fracture**

The system was evaluated using **two levels of analysis**:

1. **Slice-level prediction**
2. **Patient-level prediction**

A patient was categorized as **fractured** if **two consecutive CT slices produced a fracture probability greater than 0.99**.

### Results

The model achieved an **overall accuracy of ~80%** in detecting maxillofacial fractures.

The MFDS system helps:

- Reduce diagnostic delays
- Minimize human error
- Assist radiologists in trauma diagnosis
- Improve patient care

---

# ✨ Key Features

- Deep Learning based **fracture detection system**
- Uses **Transfer Learning for improved performance**
- **Automatic CT scan classification**
- Image preprocessing using **OpenCV**
- Binary classification: **Fracture / No Fracture**
- Designed for **clinical decision support**

---

# 🛠️ Tech Stack

| Technology |          Purpose        |
|------------|-------------------------|
|   Python   | Programming Language    |
| TensorFlow | Deep Learning Framework |
|    Keras   | Neural Network API      |
|   OpenCV   | Image preprocessing     |
|   NumPy    | Numerical computation   |
| Matplotlib | Data visualization      |

---

# 📂 Project Structure

```
Maxillofacial-Fracture-Detection
│
├── Dataset/                 # CT scan dataset
│   ├── fracture/
│   └── no_fracture/
│
├── model/                   # Saved trained models
│
├── testimages/              # Sample images used for prediction
│
├── train.py                 # Model training script
│
├── FractureDetection.py     # Fracture detection program
│
├── run.bat                  # Script to run the system
│
└── README.md
```
# 📊 Dataset

The dataset consists of **Computed Tomography (CT) scans of patients with maxillofacial trauma**.

The images are categorized into two classes:

### Fracture
CT scans showing **maxillofacial fractures**

### No Fracture
CT scans showing **no fractures**
---

### Dataset Structure
```
Dataset
│
├── fracture
│   ├── image1.png
│   ├── image2.png
│   └── ...
│
└── no_fracture
    ├── image1.png
    ├── image2.png
    └── ...
```

⚠️ **Dataset is not included in this repository due to GitHub file size limitations.**

# ⚙️ Model Architecture

The system uses **Transfer Learning with a pre-trained CNN**.

### Workflow

```
CT Scan Image
      │
      ▼
Image Preprocessing
(OpenCV)
      │
      ▼
Transfer Learning CNN
      │
      ▼
Feature Extraction
      │
      ▼
Fully Connected Layers
      │
      ▼
Binary Classification
(Fracture / No Fracture)
```
# 🧪 Training Process

Model training includes the following steps:

1. Load CT scan dataset
2. Image preprocessing
3. Dataset normalization
4. Transfer learning using a pre-trained CNN
5. Model fine-tuning
6. Validation and performance evaluation
7. Save trained model

Run training using:

```bash
python train.py
```

---

# 🔍 Running the Detection System

To detect fractures from CT scans:

```bash
python FractureDetection.py
```

Or run using the batch file:

```bash
run.bat
```

The program will process CT images and output **fracture predictions**.

---

# 📈 Model Performance

| Metric     | Value |
|------------|-------|
| Accuracy   | ~80% |
| Model Type | CNN + Transfer Learning |
| Classification | Binary |

---

# 🔬 Applications

This system can be applied in:

- Hospitals – Trauma diagnosis support  
- Medical AI systems – Automated fracture detection  
- Radiology research – AI-assisted imaging analysis  
- Emergency medicine – Faster diagnosis

---

# 🌍 Healthcare Impact

The system can help improve healthcare by:

- Supporting radiologists in trauma diagnosis
- Reducing diagnostic delays
- Improving clinical workflow
- Reducing risk of human error

---

# 🚀 Future Improvements

Possible improvements include:

- Training with larger CT scan datasets
- Implementing 3D CNN models
- Increasing diagnostic accuracy
- Integrating with hospital PACS systems
- Building a real-time medical imaging interface

---

# 📜 License

This project is licensed under the **MIT License**.

---

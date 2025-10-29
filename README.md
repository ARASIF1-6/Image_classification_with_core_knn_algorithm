# 🖼️ Image Classification using Core KNN Algorithm

A simple yet powerful **image classification system** built using the **K-Nearest Neighbors (KNN)** algorithm from scratch.
This project demonstrates classical machine learning for image recognition without deep learning — perfect for learning and research purposes.

---

## 🚀 Project Overview

This project aims to classify images (for example: animals like **Cheetah**, **Jaguar**, **Leopard**, **Lion**, **Tiger**) using the **KNN algorithm**.
Instead of using a deep neural network, we use pixel-based or histogram-based features and rely on **distance metrics** to find the closest image classes.

Key steps:

* Load and preprocess image data.
* Extract numerical features from images.
* Split data into train and test sets.
* Train and evaluate the KNN model.
* Visualize classification results.

---

## 🧠 Why KNN?

KNN is one of the simplest and most interpretable machine learning algorithms.
It works well for small to medium-sized datasets and is excellent for **understanding distance-based learning**.
Here we apply it to images to show how classical ML can still perform effectively without complex neural networks.

---

## 📂 Repository Structure

```
Image_classification_with_core_knn_algorithm/
│
├── knn.ipynb                # Jupyter Notebook for the entire process
├── data/                    # Dataset folder (Cheetah, Jaguar, Leopard, Lion, Tiger subfolders)
├── models/                  # Optional: trained KNN model or pickled classifier
├── assets/                  # Optional: result plots, confusion matrix, sample outputs
└── README.md                # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ARASIF1-6/Image_classification_with_core_knn_algorithm.git
cd Image_classification_with_core_knn_algorithm
```

### 2️⃣ Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install numpy matplotlib opencv-python scikit-learn
```

---

## 🧩 How to Run the Project

### 🔹 Option 1: Run the Notebook

```bash
jupyter notebook knn.ipynb
```

### 🔹 Option 2: Run the Script (if available)

You can convert the notebook to a `.py` script or use the example code below.

---

## 🧾 Full Example Code (Core Implementation)

Below is a **complete working example** that loads images, extracts features, trains a KNN model, and evaluates accuracy:

```python
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----- PARAMETERS -----
DATASET_PATH = "data"        # Root folder containing class subfolders
IMG_SIZE = (64, 64)          # Resize all images to 64x64
K = 3                        # Number of neighbors for KNN

# ----- DATA PREPARATION -----
data = []
labels = []

print("[INFO] Loading images from dataset...")
for label in os.listdir(DATASET_PATH):
    class_dir = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_dir):
        continue

    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        features = img.flatten()  # Flatten image to 1D vector
        data.append(features)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

print(f"[INFO] Loaded {len(data)} images across {len(np.unique(labels))} classes.")

# ----- SPLIT DATA -----
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)
print(f"[INFO] Training samples: {len(trainX)}, Testing samples: {len(testX)}")

# ----- TRAIN KNN MODEL -----
print("[INFO] Training KNN classifier...")
knn = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
knn.fit(trainX, trainY)

# ----- EVALUATE MODEL -----
print("[INFO] Evaluating model...")
predY = knn.predict(testX)
print("\nClassification Report:\n")
print(classification_report(testY, predY))

# ----- CONFUSION MATRIX -----
cm = confusion_matrix(testY, predY)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# ----- SAMPLE PREDICTIONS -----
print("[INFO] Displaying sample predictions...")
for i in range(5):
    idx = np.random.randint(0, len(testX))
    img = testX[idx].reshape(IMG_SIZE[0], IMG_SIZE[1], 3)
    plt.imshow(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {predY[idx]} | Actual: {testY[idx]}")
    plt.axis('off')
    plt.show()
```

---

## 📊 Example Output

| Input Image | Predicted Class |
| ----------- | --------------- |
| 🐆 Leopard  | Leopard         |
| 🦁 Lion     | Lion            |
| 🐅 Tiger    | Tiger           |

You’ll also get:

* A **classification report** (precision, recall, F1-score)
* A **confusion matrix heatmap**
* Random sample predictions with images and labels

---
> “Learning Machine Learning starts with understanding the simplest algorithms — KNN is one of them.” 🧠

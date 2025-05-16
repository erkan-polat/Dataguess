# 🐶🐱 Cats vs Dogs Image Classification

This project uses a binary image classification dataset containing two classes:

🐱 Cats

🐶 Dogs

The images are organized into train, validation, and test folders, and loaded using image_dataset_from_directory API in TensorFlow.

## 📁 Dataset

The dataset contains two classes:
- **Cats**: ~4000 images
- **Dogs**: ~4000 images

Data is split into `training_set`, and `test_set` sets using `data`.

## 🛠️ Data Preprocessing
The following preprocessing steps were applied to prepare the images for training:

All images resized to (224, 224) pixels

Pixel values normalized to the [0, 1] range using Rescaling(1./255)

Data loaded using TensorFlow's image_dataset_from_directory

Dataset performance improved using AUTOTUNE for parallel data loading

Preprocessing for ResNet50 using preprocess_input to match ImageNet training conditions

## 🧠 Models

### 1. Transfer Learning (ResNet50)
- Pre-trained on ImageNet
- Fine-tuned with a new classifier head

### 2. CNN (optional)
- Custom 3-layer CNN for baseline comparison

## ⚙️ Training Parameters
- Epochs: 10
- Optimizer: Adam
- Loss: Binary Crossentropy
- Batch Size: 32

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- Confusion Matrix

📊 Evaluation Metrics
To assess the model’s performance, the following metrics were used:

Metric	Description
Accuracy	Overall classification correctness
Precision	Correct positive predictions
Recall	Correctly identified positive instances
Confusion Matrix	Breakdown of predictions by class
Learning Curves	Accuracy & loss over training epochs

## 🔍 Results

| Metric     | Value  |
|------------|--------|
| Accuracy   | ~98%   |
| Precision  | 0.98   |
| Recall     | 0.97   |

Confusion matrix and training curves are included in the notebook.

## 🏭 Real-World Applications

- **Pet Identification**: Automated classification in smart pet feeders or shelters
- **Surveillance Systems**: Distinguishing between stray animals for alerting
- **Veterinary Tools**: Image-based triaging system in clinics

## 🚀 Tools Used

- TensorFlow 
- matplotlib, seaborn
- scikit-learn

## ▶️ Run the Notebook
```bash
pip install -r requirements.txt
jupyter cat_dog_classification.ipynb

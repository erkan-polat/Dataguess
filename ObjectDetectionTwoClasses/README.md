# 😷 Mask Detection Using YOLOv8 (Object Detection)

This project implements a deep learning-based object detection system to identify whether individuals are wearing face masks **correctly**, **incorrectly**, or **not at all** using the YOLOv8 architecture.

---

## 📁 Dataset

The dataset contains **853 images** with annotations in **PASCAL VOC** format, covering three categories:

* `with_mask`
* `without_mask`
* `mask_worn_incorrectly`

**Source**: [MakeML - Mask Dataset](https://makeml.app/datasets/mask)

---

## 🔧 Project Structure

```
ObjectDetection/
│
├── data/
│   ├── images/            # Raw input images
│   └── annotations/       # VOC-style XML annotation files
│
├── labels/                # YOLO-formatted output
│   ├── images/train
│   ├── images/val
│   ├── labels/train
│   └── labels/val
│
├── runs/                  # Training logs & weights (created by YOLOv8)
├── data.yaml              # YOLOv8 config file
└── mask_detection.ipynb   # Jupyter Notebook for training & inference
```

---

## 📌 Model Architecture

We use the **YOLOv8n (Nano)** variant from the [Ultralytics](https://github.com/ultralytics/ultralytics) library for:

* Speed and efficiency
* Real-time inference
* Easy training on custom datasets

---

## 🚀 Installation

```bash
pip install ultralytics
pip install opencv-python matplotlib scikit-learn pandas
```

---

## 📊 Training

* Images and annotations were converted from **VOC** to **YOLO** format.
* A `LabelEncoder` was used to convert class names into integer class IDs.
* Dataset was split: **80% for training**, **20% for validation**.
* `data.yaml` config was created dynamically.

### Training Command

```python
model.train(
    data='labels/data.yaml',
    epochs=50,
    imgsz=416,
    batch=16,
    cache=True
)
```

---

## 🧠 Inference & Visualization

After training, we use the best model to run inference and visualize the results.

```python
model = YOLO('runs/detect/train6/weights/best.pt')
results = model(img)
result_img = results[0].plot()
```

**Output includes:**

* Predicted class label
* Confidence score
* Bounding boxes drawn on image

---

## 🏠 Real-World Industrial Applications

This solution can be applied in various industrial and manufacturing settings such as:

* 🏠 **Factory floors**: Ensure compliance with mask-wearing protocols
* 🏥 **Hospitals and labs**: Real-time monitoring for health safety
* 🏢 **Smart buildings**: Automated access control based on mask usage
* 🏍️ **Retail stores**: Public safety monitoring and alerting

---

## 🏋️ Future Improvements

* Switch to larger YOLOv8 models (`s`, `m`, or `l`) for higher accuracy
* Integrate with live video stream for real-time detection
* Apply data augmentation techniques to improve generalization
* Connect with an alert system for safety enforcement

---

## 📜 License

This project uses the **MakeML Mask Dataset**, which is **Public Domain**.

---

## 🙌 Acknowledgements

* Dataset by [MakeML](https://makeml.app/datasets/mask)
* YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics)
* Project maintained and implemented in Python with PyTorch and OpenCV

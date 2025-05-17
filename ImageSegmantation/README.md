# 🧠 Brain Tumor Segmentation with U-Net

This project implements a semantic segmentation solution for medical images using a custom U-Net architecture to segment brain tumors from MRI scans.

---

## 📁 Dataset

* **Source**: Custom dataset in COCO annotation format (from Roboflow)
* **Images**: 2146 annotated brain scan images
* **Annotations**: COCO-style polygon masks
* **Classes**:

  * `0`: Non-Tumor
  * `1`: Tumor
* **Resolution**: Resized to 640x640 (during annotation) → resized to 128x128 (during model training)

---

## 🛠️ Preprocessing Pipeline

* Convert COCO polygon annotations to binary mask images (using `cv2.fillPoly`)
* Resize images and masks to (128, 128)
* Normalize pixel values to \[0, 1] range
* Convert masks to binary format (0 or 1)

---

## 📦 Project Structure

```
ImageSegmentation/
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
├── working/
│   ├── train_masks/
│   ├── val_masks/
│   └── test_masks/
├── annotations/
│   ├── train_annotation.json
│   ├── valid_annotation.json
│   └── test_annotation.json
```

---

## 🧠 Model Architecture – U-Net

* **Encoder (contracting path)**: 4 downsampling blocks with Conv2D + MaxPooling2D
* **Bottleneck**: Double Conv block with 1024 filters
* **Decoder (expanding path)**: 4 upsampling blocks with Conv2DTranspose + skip connections
* **Output**: 1-channel sigmoid activated output for binary mask prediction

---

## 🔧 Model Implementation (TensorFlow)

```python
def build_unet_model():
    inputs = layers.Input(shape=(128, 128, 3))
    ... # downsample_blocks
    ... # bottleneck
    ... # upsample_blocks
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u9)
    return tf.keras.Model(inputs, outputs)
```

---

## 🏋️ Model Training

```python
unet_model.compile(optimizer="adam", loss="BinaryCrossentropy", metrics=["accuracy"])
unet_model.fit(X_train, y_train,
               validation_data=(X_val, y_val),
               epochs=50,
               callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
```

* **Achieved Accuracy**: \~96.2%
* **EarlyStopping** used to prevent overfitting

---

## 🔍 Prediction & Visualization

```python
def predict(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    ...
    pred_mask = model.predict(input_image)
    pred_mask[pred_mask >= 0.5] = 1
    pred_mask[pred_mask < 0.5] = 0
    return pred_mask
```

* Visualized prediction alongside original image and true mask for manual evaluation

---

## 🏭 Real-World Applications

* **Medical Imaging**: Assist radiologists in identifying tumors with pixel-level accuracy
* **Surgical Planning**: Help determine tumor boundaries and growth
* **Healthcare AI**: Automate diagnosis pipelines and monitor treatment progression

---

## 🧪 Requirements

* Python 3.8+
* TensorFlow 2.x
* OpenCV
* NumPy
* Matplotlib

Install with:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

---

## 📜 License

This project uses data licensed under **CC BY 4.0**.

---

## 🙌 Acknowledgments

* Dataset via Roboflow
* U-Net paper: Ronneberger et al., 2015

# RTML_Assignment_3

# **Vision Transformer (ViT) Fine-Tuning & Inference**

This assignment involves **fine-tuning a Vision Transformer (ViT) model** on a sports classification dataset. The model is trained using a **pretrained checkpoint from epoch 7**, which achieved **70% accuracy**, and was further fine-tuned to **achieve over 94% accuracy** at epoch 30. The assignment also includes an **inference module** to classify unseen sports images.

### **Fine-Tuning Task Overview**

- The pretrained weights at **epoch 7 (70% accuracy)** can be downloaded from:
  [ViT Checkpoint (Epoch 7)](https://drive.google.com/file/d/1Fewu2rhbqw99THDGPzDMPzgSu8J5iHfO/view?usp=sharing)
- The fine-tuning process aims to **improve accuracy beyond 94%** by using **different training strategies**, such as:

## **📂 Project Files**

| **File**                | **Description**                                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `train.py`              | Script for training the ViT model on the dataset. Implements a **learning rate scheduler** and **best model saving**. |
| `inference.py`          | Runs inference on test images. Displays **actual and predicted class labels**.                                        |
| `dataset.py`            | Dataset class to **load and process** images and labels from `sports.csv`.                                            |
| `logger.py`             | Handles **logging of training metrics** to track model performance.                                                   |
| `Utils.py`              | Contains helper functions for **visualizing** images and transformations.                                             |
| `training_log.txt`      | Stores **epoch-wise training & validation losses and accuracies**.                                                    |
| `training_plots.png`    | Graph showing **training and validation accuracy over epochs**.                                                       |
| `inference_results.txt` | Stores inference results for test images.                                                                             |
| `README.md`             | This documentation file.                                                                                              |

---

## **📊 Training Results**

The model was trained for **30 epochs**, achieving over **94% accuracy**. Below is the epoch-wise performance:

| **Epoch** | **Train Loss** | **Train Accuracy** | **Learning Rate** | **Validation Loss** | **Validation Accuracy** | **Best Model Saved** | **Time Taken** |
| --------- | -------------- | ------------------ | ----------------- | ------------------- | ----------------------- | -------------------- | -------------- |
| 8         | 0.5990         | 83.95%             | 0.000029          | 0.1887              | 93.65%                  | ✅                    | 4m 41s         |
| 9         | 0.1684         | 95.59%             | 0.000027          | 0.1612              | 93.85%                  | ✅                    | 4m 39s         |
| 10        | 0.0867         | 97.58%             | 0.000024          | 0.2432              | 92.46%                  | ❌                    | 4m 36s         |
| 11        | 0.0663         | 98.05%             | 0.000020          | 0.1592              | 94.84%                  | ✅                    | 4m 42s         |
| 12        | 0.0305         | 99.23%             | 0.000015          | 0.0761              | 97.22%                  | ✅                    | 4m 35s         |
| 21        | 0.0036         | 99.90%             | 0.000010          | 0.0469              | 98.61%                  | ✅                    | 4m 34s         |
| 30        | 0.0344         | 99.08%             | 0.000024          | 0.1386              | 96.43%                  | ❌                    | 4m 34s         |

Total Training Time: **105m 30s**

---

## **🖼️ Inference Results**

Using the **best model saved at epoch 21**, we ran inference on test images. Below are the results:



**📌 Inference Images**

`dataset/test/baseball/3.jpg`

![image](https://github.com/user-attachments/assets/839ff594-3139-4410-8537-0f8144bb2d4b)

`dataset/test/archery/2.jpg`

![image](https://github.com/user-attachments/assets/be779df8-f514-471b-925e-af6bd82be897)

`dataset/test/cricket/5.jpg`

![image](https://github.com/user-attachments/assets/b698f283-8133-401b-afcf-64fca653382e)







| **Image**                     | **Actual Class** | **Predicted Class** |
| ----------------------------- | ---------------- | ------------------- |
| `dataset/test/baseball/3.jpg` | **Baseball**     | **Baseball** ✅      |
| `dataset/test/archery/2.jpg`  | **Archery**      | **Archery** ✅       |
| `dataset/test/cricket/5.jpg`  | **Cricket**      | **Cricket** ✅       |

###

\
\


---

## **🛠️ Modifications & Improvements**

### **1️⃣ Learning Rate Scheduler**

- Implemented **Cosine Annealing LR Scheduler** to dynamically adjust learning rate.
- Helps prevent overfitting and improves model generalization.
- The LR decreases over time and cycles back up periodically.

### **2️⃣ Automatic Best Model Saving**

- The best model (highest validation accuracy) is **automatically saved**.
- Best model: **Epoch 21 (98.61% Accuracy)**.

### **3️⃣ Data Augmentations for Generalization**

To improve generalization, the following **data augmentation techniques** were applied:

- **Random Horizontal Flip (50%)** - Helps the model generalize to mirrored versions of the images.
- **Color Jitter (Brightness & Contrast)** - Introduces minor variations in lighting conditions to enhance robustness.
- **Random Rotation (±15 degrees)** - Helps the model learn invariance to different orientations.
- **Center Cropping & Resizing** - Ensures consistent input size while focusing on the most informative part of the image.

### **4️⃣ Logging & Visualization**

- Training results are stored in **`training_log.txt`**, which logs epoch-wise losses, accuracy, and learning rate updates.
- **`training_plots.png`** provides a visual representation of training and validation accuracy over epochs.
- **Inference results** are recorded in **`inference_results.txt`**, showing actual and predicted labels for test images.
- The logging system helps in tracking performance improvements and identifying areas for further fine-tuning.



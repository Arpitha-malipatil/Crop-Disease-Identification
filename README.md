# 🌿 Plant Disease Identification using Machine Learning

A machine learning project that identifies plant leaf diseases using **SVM**, **Random Forest**, and a **CNN** trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) (38 classes, ~54,000 images).

> **Memory-safe pipeline** — uses `ImageDataGenerator.flow_from_directory()` for CNN training and a 5,000-sample draw for SVM/RF. No full dataset is ever loaded into RAM.

---

## 👥 Team

| # | Notebook | Step | Member |
|---|----------|------|--------|
| 1 | `step01_data_cleaning_Tanusha.ipynb` | Data Cleaning | Tanusha |
| 2 | `step02_data_validation_Zaina.ipynb` | Data Validation | Zaina |
| 3 | `step03_data_augmentation_Vishwa.ipynb` | Data Augmentation | Vishwa |
| 4 | `step04_feature_engineering_Rakesh.ipynb` | Feature Engineering | Rakesh |
| 5 | `step05_train_test_split_Siri.ipynb` | Train / Test Split | Siri |
| 6 | `step06_feature_processing_Ranjith.ipynb` | Feature Processing (PCA) | Ranjith |
| 7 | `step07_model_training_Vaseem.ipynb` | Model Training (SVM · RF · CNN) | Vaseem |
| 8 | `step08_model_evaluation_Rudresh.ipynb` | Model Evaluation | Rudresh |
| 9 | `step09_model_architecture_Shankar.ipynb` | Architecture & Visualisation | Shankar |
| 10 | `step10_frontend_deployment_Sneha.ipynb` + `app.py` | Frontend Deployment | Sneha |
| 11 | ` Documentation | Pavan C |

**Guide:** Sumethra Devi Ma'am  
**Institution:** Dayananda Sagar Academy of Technology and Management

---

## 📁 Dataset

- **Source:** [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Classes:** 38 plant–condition combinations
- **Structure:** `color/<ClassName>/<image>.jpg`

> ⚠️ **Before running any notebook**, update the `dataset_path` variable to point to your local `color/` folder:
> ```python
> dataset_path = r"C:\your\path\to\plantvillage\color"
> ```

---

## 🚀 Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-org>/plant-disease-identification.git
cd plant-disease-identification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and extract so the path looks like:
```
plantvillage/
└── color/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    └── ...  (38 folders total)
```

---

## ▶️ How to Run

### Option A — Run each member's notebook individually

Each notebook is **self-contained** — open it in Jupyter or Google Colab, update `dataset_path`, and run all cells top to bottom.

```
Step 1  →  step01_data_cleaning_Tanusha.ipynb
Step 2  →  step02_data_validation_Zaina.ipynb
Step 3  →  step03_data_augmentation_Vishwa.ipynb
Step 4  →  step04_feature_engineering_Rakesh.ipynb
Step 5  →  step05_train_test_split_Siri.ipynb
Step 6  →  step06_feature_processing_Ranjith.ipynb
Step 7  →  step07_model_training_Vaseem.ipynb       ← trains all 3 models
Step 8  →  step08_model_evaluation_Rudresh.ipynb
Step 9  →  step09_model_architecture_Shankar.ipynb
Step 10 →  step10_frontend_deployment_Sneha.ipynb   ← saves model files
```

> Steps 7–10 re-run the earlier setup internally, so they work as standalone files too.

### Option B — Run the Streamlit frontend

After running Step 10 (which saves the model files), launch the UI:

```bash
streamlit run app.py
```

Make sure `plant_disease_cnn.h5` and `pca_encoder.pkl` are in the same folder as `app.py`.

---

## 💾 Generated Model Files

These are created by **Step 10 (Sneha)** and used by `app.py`:

| File | Description |
|------|-------------|
| `plant_disease_cnn.h5` | Trained CNN model (Keras) |
| `svm_model.pkl` | Trained SVM classifier |
| `rf_model.pkl` | Trained Random Forest classifier |
| `pca_encoder.pkl` | PCA transformer + LabelEncoder (dict) |

---

## 🏗️ CNN Architecture

```
Input: 32 × 32 × 3 (RGB)
├── Conv2D(32, 3×3, relu) → BatchNorm → MaxPool(2×2)
├── Conv2D(64, 3×3, relu) → BatchNorm → MaxPool(2×2)
├── Conv2D(128, 3×3, relu) → MaxPool(2×2)
├── Flatten → Dense(256, relu) → Dropout(0.4)
└── Dense(38, softmax)

Optimizer : Adam
Loss      : Categorical Crossentropy
Epochs    : 10
```

---

## 📊 Results

| Model | Training Data | Accuracy |
|-------|--------------|----------|
| SVM (linear kernel) | 5,000-sample PCA subset | ~70% |
| Random Forest (100 trees) | 5,000-sample PCA subset | ~66% |
| **CNN** | **Full dataset via generator** | **~87%** |

---

## 📦 Repository Structure

```
plant-disease-identification/
│
├── step01_data_cleaning_Tanusha.ipynb
├── step02_data_validation_Zaina.ipynb
├── step03_data_augmentation_Vishwa.ipynb
├── step04_feature_engineering_Rakesh.ipynb
├── step05_train_test_split_Siri.ipynb
├── step06_feature_processing_Ranjith.ipynb
├── step07_model_training_Vaseem.ipynb
├── step08_model_evaluation_Rudresh.ipynb
├── step09_model_architecture_Shankar.ipynb
├── step10_frontend_deployment_Sneha.ipynb
│
├── app.py                  ← Streamlit frontend (Sneha)
├── requirements.txt
└── README.md
```

---

## 🛡️ Notes

- All notebooks include **full imports** so they open independently in Jupyter or Google Colab.
- The pipeline is **memory-safe** — no `MemoryError` on standard laptops.
- `pca_encoder.pkl` stores both the PCA transformer and LabelEncoder as a dict: `{'pca': ..., 'encoder': ...}`.

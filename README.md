# Skin vs. Non-Skin Classification

A deep learning project to classify images as "Skin" or "Non-Skin" using a ResNet18 model and the Pratheepan Skin Dataset.

## 📌 Features
- **Data Pipeline**: Automated patch extraction from images and masks.
- **Model**: ResNet18 with Transfer Learning.
- **Interactive App**: Streamlit web interface for easy inference.
- **Evaluation**: Detailed metrics (Accuracy, Precision, Recall, F1) and Confusion Matrix.

## 📂 Project Structure
```
.
├── data/               # Dataset (Images & Masks)
├── outputs/            # Trained Models & Metrics
├── src/
│   ├── app.py          # Streamlit Web App
│   ├── eval.py         # Evaluation Script
│   ├── make_patches.py # Data Preprocessing
│   ├── predict.py      # CLI Prediction Script
│   └── train.py        # Training Script
├── requirements.txt    # Dependencies
└── README.md
```

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd skin_non_skin_project
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Dataset
This project uses the **Pratheepan Skin Dataset**.
1.  Download the dataset from [here](http://cs-chan.com/downloads_skin_dataset.html).
2.  Structure your `data` folder as follows:
    ```
    data/
    └── raw/
        ├── images/   # All original images
        └── masks/    # Corresponding binary masks
    ```

## 🛠 Usage

### 1. Data Preparation
Generate training patches from the raw images:
```bash
python src/make_patches.py --images_dir data/raw/images --masks_dir data/raw/masks --out_dir data/patches
```

### 2. Training
Train the ResNet18 model:
```bash
python src/train.py --epochs 8
```
*The model will be saved to `outputs/model.pt`.*

### 3. Evaluation
Generate metrics and confusion matrix:
```bash
python src/eval.py
```
Check `outputs/` for `metrics.json` and `confusion_matrix.png`.

### 4. Running the App
Start the Streamlit interface to test with your own images:
```bash
streamlit run src/app.py
```

## 📈 Results
The model achieves high accuracy on the test set. Detailed metrics are saved in `outputs/metrics.json` after evaluation.


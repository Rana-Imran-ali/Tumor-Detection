# NeuroScan AI: Brain Tumor MRI Classification System

NeuroScan AI is a deep learning and computer vision web application designed to assist in classifying brain MRI scans into four distinct categories:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor (Healthy)**

Powered by **Transfer Learning** using **MobileNetV2** (TensorFlow/Keras), this system features a modular **Flask** backend, a modern **Bootstrap 5** frontend styled with a **professional navy-blue and medical-teal theme**, **Chart.js** real-time visualizers, and **SQLite** for persisting scan records and diagnostic logs.

---

## 📁 Project Structure

```text
Tumor Detection/
│
├── app.py                         # Flask web application routes & API endpoints
├── database.py                    # SQLite database helper (CRUD operations for scan logs)
├── model.py                       # MobileNetV2 architecture, preprocessing & real neural inference
├── train.py                       # Training pipeline using transfer learning (MobileNetV2)
├── evaluate.py                    # Held-out test evaluation generating confusion matrix & metrics
├── build_baseline_model.py        # Generates baseline MobileNetV2 model with real neural weights
├── generate_performance_graphs.py # Generates high-res medical navy & teal performance charts
├── generate_sample_scans.py       # Generates synthetic demonstration brain MRI scans
├── predict_sample.py              # Standalone CLI prediction tool for single MRI scans
├── download_dataset.py            # Automated Kaggle dataset extractor & downloader
├── test_app.py                    # Automated test suite (8 tests covering all routes & inference)
├── requirements.txt               # Project dependencies
├── README.md                      # Comprehensive project documentation
├── neuroscan.db                   # SQLite database (auto-created on first run)
│
├── models/
│   ├── brain_tumor_model.keras    # Trained MobileNetV2 model checkpoint (10.6 MB)
│   └── class_names.json           # Class label mapping array
│
├── outputs/
│   ├── confusion_matrix.png       # 4x4 Confusion Matrix heatmap
│   ├── training_curves.png        # Training vs. validation accuracy and loss curves
│   ├── class_metrics_bar.png      # Per-class precision, recall, and F1-score chart
│   └── metrics.json               # Quantitative benchmark metrics
│
├── static/
│   ├── css/
│   │   └── style.css              # Custom Navy-Blue & Teal medical design system
│   ├── js/
│   │   └── main.js                # Drag-and-drop, Chart.js visualizer, AJAX inference
│   ├── img/                       # Performance graphs displayed in the web UI
│   ├── uploads/                   # Uploaded patient MRI scans (UUID prefixed)
│   └── samples/                   # Built-in demo MRI scans for 1-click testing
│
└── templates/
    ├── base.html                  # Base layout with navbar, footer, Chart.js & disclaimer
    ├── index.html                 # Diagnostic workstation dashboard & live analyzer
    ├── performance.html           # Model evaluation graphs & quantitative benchmarks
    ├── history.html               # Patient scan history table with search, filter & CSV export
    └── about.html                 # Clinical pathology reference & architecture guide
```

---

## 🚀 Exact Commands to Setup & Run

### 1. Install Dependencies
Open PowerShell or your preferred terminal in the project directory:

```powershell
# Navigate to project folder
cd "c:\Users\Eventbuizz\Desktop\Tumor Detection"

# Create a virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install all requirements
pip install -r requirements.txt
```

*(Note: TensorFlow, Flask, Pillow, NumPy, Scikit-Learn, Matplotlib, and Requests will be installed.)*

---

### 2. Model Training & Preparation

#### Option A: Quick Baseline Initialization (Immediate, Out-of-the-Box)
If you want to train and initialize the MobileNetV2 classification head immediately without downloading the full 150 MB dataset:
```powershell
python build_baseline_model.py
```
This fits the classification head on multi-class scan variations and saves the functional weights to `models/brain_tumor_model.keras`.

#### Option B: Full Kaggle Dataset Training Pipeline
To train on the official Kaggle Brain Tumor MRI Dataset (7,023 images):
1. **Download the dataset**:
   ```powershell
   python download_dataset.py
   ```
   *(Or download `archive.zip` from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and extract into `dataset/`)*
2. **Execute training**:
   ```powershell
   python train.py --data_dir dataset/ --epochs 15 --batch_size 32
   ```

---

### 3. Generate Evaluation Graphs & Benchmarks
To generate or refresh the high-resolution navy-blue and teal styled confusion matrix, training curves, and per-class metrics:
```powershell
python generate_performance_graphs.py
```
Outputs will be written to `outputs/` and `static/img/`.

---

### 4. Run Automated Test Suite
To verify model loading, preprocessing parity, inference, file security, SQLite persistence, and CSV export:
```powershell
python test_app.py
```
Expected output:
```text
Ran 8 tests in ~29s
OK
```

---

### 5. Launch the Web Application
Start the Flask development server:
```powershell
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🔮 Standalone CLI Inference
You can also run predictions directly from the command line without launching the web server:
```powershell
python predict_sample.py --image "static/samples/meningioma_sample.jpg"
```
Example Output:
```text
=================================================================
 NeuroScan AI: Brain MRI Inference CLI
=================================================================
 Target Image : static/samples/meningioma_sample.jpg
 Model Loaded : models/brain_tumor_model.keras
-----------------------------------------------------------------
 PREDICTION RESULT: MENINGIOMA
 CONFIDENCE SCORE : 87.14%
-----------------------------------------------------------------
 Softmax Class Probabilities:
   - Glioma      :   0.34% | 
   - Meningioma  :  87.14% | ################################## <== (Highest)
   - No Tumor    :  10.79% | ####
   - Pituitary   :   1.74% | 
=================================================================
```

---

## 🌟 Key Features

1. **Professional Navy-Blue & Teal Theme**:
   - Modern medical workstation aesthetics with frosted glass navbar, HUD scanning brackets, glowing reticles, and clear typography.
2. **Real Deep Learning Inference**:
   - Genuine MobileNetV2 neural inference running on 224x224 RGB inputs normalized to `[-1.0, +1.0]`.
   - Never uses hardcoded predictions.
3. **Four-Class Probability Breakdown**:
   - Color-coded progress bars and real-time interactive **Chart.js** bar visualizer displaying the Softmax probability distribution across all 4 categories.
4. **Clinical Findings & Radiological Hallmarks**:
   - Displays specific tumor hallmarks (infiltrative edema, dural tail, sellar expansion) and recommended next clinical protocols.
5. **Secure File Handling & Validation**:
   - Enforces 16MB file limit, extension whitelist (`.jpg`, `.jpeg`, `.png`, `.webp`), path traversal protection (`is_safe_path`), and binary image byte verification via PIL.
6. **SQLite Prediction History**:
   - Automatically logs patient ID, filename, primary prediction, confidence, JSON probabilities, and timestamp to `neuroscan.db`.
7. **CSV Export**:
   - One-click export (`/history/export`) generating structured CSV logs complete with multiclass probability breakdowns and medical disclaimers.
8. **Empirical Model Performance Page**:
   - Dedicated dashboard at `/performance` displaying 4 KPI summary cards, training curves, confusion matrix, and held-out test matrix.
9. **Prominent Medical Disclaimer**:
   - Displayed in the header banner, footer, API JSON responses, and CSV exports to ensure responsible clinical support protocol.
# Tumor-Detection

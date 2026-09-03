# 🚀 Deploying Pneumonia Diagnostic Hub on Streamlit

This guide provides complete, step-by-step instructions for deploying the **Pneumonia Diagnostic Hub • AI Radiology Workstation** on **Streamlit Community Cloud**, as well as running it locally or via Docker.

---

## 🌟 Streamlit Workstation Overview

The Streamlit interface (`streamlit_app.py`) provides an enterprise-grade Clinical Decision Support System:
- 🫁 **4-Model Weighted Soft-Voting Consensus Engine** (MobileNetV2, ResNet50, EfficientNetB0, VGG19)
- 🔬 **Interactive Explainable AI (Grad-CAM)** with real-time colormap selection and parenchymal opacity blending
- 📁 **Medical DICOM (.dcm) Ingestion** with full header parsing, acquisition tags, and VOI LUT windowing
- 🧪 **Pre-Loaded Clinical Benchmark Cases** (Normal CXR, Bacterial Lobar, Viral Interstitial)
- 🏛️ **Multi-Architecture Comparison Grid** with latency and isolated attention heatmaps
- 📄 **Publication-Grade Clinical PDF Report Export** with radiologist notes and 1-click download

---

## ☁️ Method 1: Deploy to Streamlit Community Cloud (Free Hosting)

Streamlit Community Cloud allows free deployment directly connected to your GitHub repository.

### Step 1: Ensure Repository is Pushed to GitHub
Make sure your changes are pushed to your GitHub repository:
```bash
git add .
git commit -m "feat: add Streamlit radiology workstation and deployment configs"
git push origin main
```

### Step 2: Sign in to Streamlit Community Cloud
1. Open [share.streamlit.io](https://share.streamlit.io/) in your browser.
2. Sign in with your **GitHub account**.

### Step 3: Deploy New App
1. Click the **"New app"** button.
2. Select your repository:
   - **Repository:** `Shahab-khan396/-Pneumonia-Diagnostic-Hub-AI-Radiology-Workstation-Multi-Model-Consensus-Engine-DICOM-Ready-`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
3. Click **"Deploy!"**.

Streamlit Community Cloud will automatically:
- Detect and install Linux system libraries from `packages.txt` (`libgl1`, `libglib2.0-0`, `libgomp1`).
- Install Python dependencies from `requirements.txt`.
- Apply custom clinical dark theme settings from `.streamlit/config.toml`.
- Launch the workstation live!

---

## 🧠 Handling Deep Learning Models & Git LFS on Streamlit Cloud

### GitHub File Size Limits & Git LFS
GitHub limits non-LFS file uploads to **100 MB**:
| Model Backbone | Parameters | File Size | Git Handling |
| :--- | :--- | :--- | :--- |
| **MobileNetV2** | 3.5M | ~11.5 MB | Standard Git |
| **EfficientNetB0** | 5.3M | ~20.8 MB | Standard Git |
| **ResNet50** | 25.6M | ~101.2 MB | Tracked via Git LFS |
| **VGG19** | 63.1M | ~443.6 MB | Tracked via Git LFS |

> [!TIP]
> **Streamlit Cloud RAM Optimization:**
> Streamlit Community Cloud free tier provides approximately **1 GB to 2.7 GB** of RAM.
> In `streamlit_app.py`, use the **"⚡ Streamlit Cloud Fast Mode"** toggle in the sidebar to run the consensus across MobileNetV2, ResNet50, and EfficientNetB0 while bypassing the memory-heavy VGG19 (443 MB) model.

---

## 💻 Method 2: Local Execution

To run the Streamlit Radiology Workstation on your local machine:

### 1. Activate Environment & Install Dependencies
```bash
# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Launch Streamlit Workstation
```bash
streamlit run streamlit_app.py
```
The workstation will automatically open in your default browser at `http://localhost:8501`.

---

## 🐳 Method 3: Containerized Deployment (Docker)

To run the Streamlit app inside a production Docker container:

### Build Container
```bash
docker build -t pneumonia-streamlit -f Dockerfile.streamlit .
```

### Run Container
```bash
docker run -d -p 8501:8501 --name pneumonia-hub-streamlit pneumonia-streamlit
```
Access the application at `http://localhost:8501`.

---

## ⚙️ Configuration Files Reference

| File | Purpose |
| :--- | :--- |
| `streamlit_app.py` | Main Streamlit application entry point |
| `.streamlit/config.toml` | Streamlit theme (clinical dark mode) and server limits (`maxUploadSize = 35`) |
| `requirements.txt` | Python packages (`streamlit>=1.35.0`, `tensorflow`, `pydicom`, `reportlab`) |
| `packages.txt` | Linux system dependencies for headless OpenCV (`libgl1`, `libglib2.0-0`) |
| `core/` | Modular inference, consensus, DICOM parser, and report generator engine |

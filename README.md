
# Cooperative Perception for Autonomous Vehicles using LiDAR and LLM Integration

This project focuses on enhancing perception in autonomous vehicles through **cooperative LiDAR-based object detection** and introduces a future extension with **Large Language Models (LLMs)** for emergency decision-making. The implementation leverages the **TUMTraf V2X Cooperative Perception Dataset** and builds a LiDAR-only 3D object detection pipeline using a custom-built **PointPillar model**, trained and evaluated from scratch.

## 🚘 Project Highlights

- **LiDAR-only 3D Object Detection** using PointPillar architecture
- Supports **Vehicle-only**, **Infrastructure-only**, and **Cooperative (Fused)** configurations
- Custom **data preprocessing** pipeline to handle the multi-agent TUMTraf dataset
- Bounding box **visualization in BEV (Bird’s Eye View)** using both Matplotlib and Open3D

## 📁 Folder Structure

```
├── data_utils/             # All data processing and dataset utils
│   ├── label_parser.py
│   ├── loss.py
│   ├── pointpillar.py
│   ├── target_assigner.py
│   ├── tumtraf_dataset.py
│   └── voxelizer.py
├── datasets/               # Dataset descriptor files
│   └── dataset.txt
├── models/                 # Model definition
│   └── model.py
├── output/                 # Outputs (visualizations and evaluation)
│   ├── LLM/                # Sample visuals from LLM integration
│   ├── bev_comparisons/    # GT vs Pred visualizations
│   └── train_loss_curve.png
├── preprocessing/          # Dataset preprocessing
│   └── preprocess.py
├── training/               # Training scripts
│   └── lidar_train.py
├── requirements.txt
```

## 📦 Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# (Optional) Create virtual environment
python -m venv coopenv
source coopenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Running the Project

### 1. Preprocess the Dataset

```bash
python preprocessing/preprocess.py
```

### 2. Train the LiDAR-Only PointPillar Model

```bash
python training/lidar_train.py
```

### 3. Visualize Predictions

```bash
# Output in output/bev_comparisons/
python visualize_bev.py
```

## 🔍 LLM Module (Early Integration Phase)

An experimental **LLM-based emergency decision-making module** is prepared for future integration. It takes in cooperative perception scene descriptions and outputs risk assessments or recommended actions (e.g., "slow down", "brake").

This module is structured and located under `output/LLM/`, and will serve as a decision layer downstream of perception.

## 📈 Sample Results

- **Vehicle-only LiDAR**: 80.1 mAP₃D Avg
- **Infra-only LiDAR**: 84.88 mAP₃D Avg
- **Cooperative LiDAR**: **90.76 mAP₃D Avg**

Visual samples are included in the repo under `output/bev_comparisons`.

## 📌 Future Work

- Extend model to handle **camera-only** and **camera+LiDAR** setups
- Full integration with **LLM-based risk reasoning and planning**
- Explore deployment on CARLA for sim-to-real transfer

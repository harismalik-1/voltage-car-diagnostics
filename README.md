# Voltage: Automated Vehicle Diagnostics

![UI Screenshot](./docs/homepage.png)

## Overview
Voltage is an automated vehicle diagnostics platform for connected and autonomous vehicles (CAVs). It provides real-time health monitoring, anomaly detection, and system insights using advanced machine learning models and a modern web interface.

---

## Frontend

- **Technology:** React (TypeScript), TailwindCSS, Vite, Lucide Icons
- **Why this stack?**
  - **React** offers a component-based architecture, fast rendering, and a rich ecosystem for building interactive UIs.
  - **TypeScript** provides type safety and better maintainability for large codebases.
  - **TailwindCSS** enables rapid, consistent, and modern UI styling.
  - **Vite** ensures fast development and hot-reloading.
- **Key Features:**
  - Real-time vehicle status and diagnostics
  - 3D vehicle visualization
  - File upload and download for CAN data
  - Anomaly results display

### Running the Frontend Locally
```bash
cd frontend
npm install
npm run dev
```
The app will be available at `http://localhost:8080` (or as specified by Vite).

---

## Backend

This project uses **two backend servers**:

### 1. CAN Data Server (Express.js)
- **Purpose:**
  - Serves CAN data files (e.g., `can_data.csv`) to the frontend
  - Handles log registration and metadata storage
- **Why Express?**
  - Lightweight, easy to set up for REST APIs and static file serving
  - Integrates easily with databases (e.g., PostgreSQL)
- **How to run:**
```bash
cd backend/can_data_server
npm install
node app.js
```
The server will run at `http://localhost:3000`.

### 2. ML Evaluation Server (FastAPI or similar)
- **Purpose:**
  - Receives uploaded CAN data, runs LSTM-based anomaly detection, and returns results
- **Why a separate ML server?**
  - Keeps ML dependencies isolated from the main API
  - Enables scaling and independent deployment of the ML service
- **How to run:**
```bash
cd backend/ml_eval_server
# (activate your Python environment)
uvicorn server_main:app --reload --port 8000
```
The server will run at `http://localhost:8000`.

---

## Machine Learning: LSTM Autoencoder

- **Model:** LSTM-based autoencoder for time-series anomaly detection on CAN bus data
- **Why LSTM?**
  - LSTMs (Long Short-Term Memory networks) are well-suited for sequential data like CAN logs, capturing temporal dependencies and patterns that simpler models miss.
  - Autoencoders learn to reconstruct normal sequences; high reconstruction error signals an anomaly.
- **Advantages:**
  - Handles variable-length sequences and long-term dependencies
  - Robust to noise and missing data
  - Outperforms traditional methods (e.g., thresholding, simple statistics) on complex, real-world CAN data
- **How to train/run:**
  - See `ml/train_lstm.py` for training
  - See `ml/evaluate_lstm.py` and `backend/ml_eval_server/server_main.py` for evaluation and serving

---

## Research Context: Systematic Review on Anomaly Detection in CAVs

This project is inspired by and aligned with the findings of the systematic review:
- [Systematic Review: Anomaly Detection in Connected and Autonomous Vehicles (arXiv:2405.02731v1)](https://arxiv.org/html/2405.02731v1)

**Key takeaways from the paper:**
- LSTM, CNN, and autoencoder models are the most common and effective AI methods for anomaly detection in CAVs
- Real-world CAN data is crucial for training robust models
- Multiple evaluation metrics (accuracy, precision, recall, F1-score) should be used for comprehensive assessment
- There is a need for open-source models and benchmarking datasets
- The field is moving towards real-time, on-vehicle deployment of anomaly detection systems

**How this project addresses these points:**
- Uses an LSTM autoencoder for anomaly detection
- Trains and evaluates on real CAN data
- Provides a modern, open-source stack for both research and practical deployment
- Designed for extensibility and benchmarking

---

## How to Contribute
- Fork the repo, open issues, and submit pull requests!
- See the code comments and docstrings for guidance on extending the ML models or API endpoints.

---

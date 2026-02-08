# 🕵️‍♂️ CryptoGraph: AML Detection via Graph Neural Networks

A specialized forensic pipeline designed to identify illicit Bitcoin transactions within the **Elliptic Data Set** using Graph Convolutional Networks (GCNs).

## 🌐 Live Deployment
**[🚀 Launch Investigator Dashboard](https://bitcoin-forensics-gnn-c8rajqq8njarbyovukhdb7.streamlit.app/)**

*Analyst Note: The live deployment utilizes a forensic data sample for demonstration. Full-scale inference on 203,769 transactions is performed in the containerized production environment.*

## 🏛️ Project Architecture
This project implements a multi-stage forensic audit to detect money laundering rings by analyzing topological fund flows.

### 1. Forensic Pipeline Flow
* **Forensic Feature Analysis:** Audited class imbalance (2.23% illicit) and performed statistical profiling of node features.
* **Topology Visualization:** Mapped spatial relationships of illicit nodes to identify potential laundering clusters.
* **GNN Inference Engine:** Implemented a **3-layer Graph Convolutional Network (GCN)** with **Weighted Cross-Entropy** to prioritize topological recall.
* **Performance:** Achieved a **75% Macro-average Recall**, ensuring forensic accuracy across highly imbalanced criminal transaction data.

## 🐳 MLOps & Containerization (Docker Guide)
To ensure environment parity and scalability, this project is fully containerized. Follow these steps to deploy the engine locally:

### 1. Build the Image
```bash
docker build -t bitcoin-forensics-app .
```

### 2. Run the Container
```bash
docker run -p 8501:8501 bitcoin-forensics-app

The dashboard will be accessible at http://localhost:8501.
```
### 🛠️ Local Execution (Non-Docker)
```bash
Environment Setup: pip install -r requirements.txt

Automated Pipeline: python scripts/main.py

Investigator Dashboard: streamlit run scripts/investigator_dashboard.py
```
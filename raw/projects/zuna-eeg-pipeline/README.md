## 🚀 Getting Started

This project provides a GPU-accelerated container environment for evaluating the Zuna 380M-parameter EEG foundation model. It processes the Wakeman and Henson simultaneous MEG/EEG dataset (`ds000117`), drops 25% of the EEG channels, and benchmarks Zuna's deep learning imputation against standard MNE spherical spline interpolation.

### Prerequisites
* **NVIDIA AI Workbench:** Installed on your local machine or remote compute node.
* **Hardware:** An NVIDIA GPU with sufficient VRAM for a 380M-parameter diffusion model.
* **Dataset:** The `ds000117` BIDS dataset downloaded locally.

### 1. Configure the Dataset Mount
For the container to access the EEG files without copying massive amounts of data, you must configure a Host Mount in AI Workbench.
1. Open this project in the NVIDIA AI Workbench desktop app.
2. Navigate to the **Environment** tab.
3. Under **Mounts**, add a new Host Mount:
   * **Source:** `/path/to/your/local/ds000117` *(Change this to where the data lives on your machine/node)*
   * **Target:** `/data/ds000117` *(Do not change this; the Python scripts expect this exact path)*

### 2. Build and Launch the Environment
1. Click **Build Environment** in the top right corner of AI Workbench. This will pull the PyTorch base, install all dependencies (`mne`, `mne-bids`, `zuna`), and configure CUDA passthrough.
2. Once the build is complete and the status turns green, click **Open JupyterLab**.

### 3. Run the Evaluation Batch
All preprocessing, chunking, diffusion inference, and MNE topomap generation is handled automatically by the batch script.

1. Open a **Terminal** inside JupyterLab.
2. Run the overnight evaluation pipeline:
   ```bash
   python zuna_batch_eval.py
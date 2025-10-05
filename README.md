# EEGMaSL: Code Repository

> Official codebase for EEGMaSL: Frequency-based Self-Supervised Learning for Large-Scale EEG Representation

---

## 🚀 Overview

EEGMaSL provides a flexible and efficient framework for self-supervised EEG representation learning based on frequency-domain data augmentation and the Mamba2 backbone. This repository includes scripts for data preprocessing, model training, evaluation, and visualization. The code is designed to support research in robust EEG feature learning and is adaptable for various downstream classification tasks.
<p align="center">
  <img src="MaSL/scripts/EEG_masl (2).png" alt="Overview" width="480">
</p>

## 📦 Features

- Frequency-based augmentation preserving EEG’s spectral characteristics
- Contrastive pre-training pipeline built on the Mamba2 model
- End-to-end EEG processing, from raw data to evaluation
- Flexible interface for custom datasets and tasks

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/EEGMaSL/code_for_EEGMaSL.git
cd code_for_EEGMaSL
pip install -r requirements.txt
```

# Usage Example

### MaSL pre-train

We use data after hybrid frequency processing for pre-training:
```bash
bash scripts/run_unsup_pretrain.sh
```
### MaSL fine-tuning

After modifying the path for saving the pre-trained model, the fine-tuning is also launched using the script：
```bash
bash scripts/run_sup_pretrain.sh
```

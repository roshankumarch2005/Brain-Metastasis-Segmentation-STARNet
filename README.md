# Brain Tumor & Metastasis Segmentation on MRI (STAR-Net)
**BTP Phase 1 | Department of Electrical Engineering, IIT Tirupati**

---

## 1. Welcome & Where to Start

If you are a new researcher, student, or collaborator taking over this repository, **start here first**:

1. **Read the Master Presentation:**  
   Open **[`BTP_Phase1_Master_Presentation.pdf`](file:///mnt/DATA/EE23B009/EE23B009_BTP/BTP_Phase1_Master_Presentation.pdf)** (or the PowerPoint source **[`BTP_Phase1_Master_Presentation.pptx`](file:///mnt/DATA/EE23B009/EE23B009_BTP/BTP_Phase1_Master_Presentation.pptx)**).  
   *This 29-slide visual presentation summarizes the entire project:* clinical motivation (stereotactic radiosurgery vs. whole-brain radiation), the 15-model ablation matrix, architectural innovations (STAR-Net), the discovery of clinical task asymmetry, and quantitative comparisons against literature benchmarks.

2. **Read the Master Written Report:**  
   Review **[`BTP_Final_Report.md`](file:///mnt/DATA/EE23B009/EE23B009_BTP/BTP_Final_Report.md)** and the publication-format **[`BTP_IEEE_Report.pdf`](file:///mnt/DATA/EE23B009/EE23B009_BTP/BTP_IEEE_Report.pdf)** for complete mathematical derivations, loss formulations, training dynamics, and tabular results.

3. **Check Literature Benchmarks:**  
   See **[`Literature_vs_Our_BTP_Results_Comparison.md`](file:///mnt/DATA/EE23B009/EE23B009_BTP/Literature_vs_Our_BTP_Results_Comparison.md)** and **[`Literature_and_All_Configurations_Master_Benchmark.pdf`](file:///mnt/DATA/EE23B009/EE23B009_BTP/Literature_and_All_Configurations_Master_Benchmark.pdf)** for direct baseline and prior-art comparisons.

---

## 2. Repository Directory Structure

```
EE23B009_BTP/
├── README.md                                        # This file
├── BTP_Phase1_Master_Presentation.pdf               # Master 29-slide presentation (PDF)
├── BTP_Phase1_Master_Presentation.pptx              # Master 29-slide presentation (PowerPoint)
├── BTP_Final_Report.md                              # Comprehensive technical report
├── BTP_IEEE_Report.pdf                              # IEEE format manuscript
├── Literature_vs_Our_BTP_Results_Comparison.md     # In-depth literature vs BTP benchmarks
├── Literature_and_All_Configurations_Master_Benchmark.pdf
├── STAR_Net_Overleaf_Beamer.zip                     # LaTeX Beamer slide deck package
├── build_visual_master_presentation.py              # Script that generates the PPTX presentation
├── render_latex_equations.py                        # Script that renders LaTeX math into PNGs
│
├── src/                                             # Core source code & scripts
│   ├── models/                                      # PyTorch model definitions & network blocks
│   ├── losses/                                      # Loss functions (BAA, SIAM, hinge, curriculum)
│   ├── nnUNetTrainer_*.py                           # Custom nnU-Net training classes
│   ├── evaluate_*.py                                # Model evaluation & metric calculation scripts
│   ├── prepare_*.py                                 # Dataset formatting & conversion scripts
│   └── visualize_*.py                               # Qualitative slice plotting routines
│
├── metrics_csv/                                     # Quantitative evaluation spreadsheets
│   └── *.csv                                        # 25 benchmark and per-case test metric tables
│
├── results/                                         # Project visual assets & inference outputs
│   ├── presentation_figures/                        # Visual figures used in the master slides
│   ├── architecture_diagrams/                       # High-res SVG and PNG architecture schematics
│   ├── predictions_test/                            # 30-case NIfTI test segmentation masks (.nii.gz)
│   └── qualitative_candidates/                      # Selected axial comparison slices
│
├── training_logs/                                   # Epoch-by-epoch training logs (loss curves, Dice)
├── sweep_logs/                                      # Hyperparameter grid search logs (M10 sweeps)
├── overleaf_beamer/                                 # Uncompressed LaTeX Beamer source files
│
├── nnunet_data/                                     # nnU-Net dataset & checkpoint hierarchy
│   ├── nnUNet_raw/                                  # Raw formatted NIfTI datasets (Dataset100, etc.)
│   ├── nnUNet_preprocessed/                         # Preprocessed .npy volumes and plans.json
│   └── nnUNet_results/                              # Trained model weights (.pth) & checkpoints
│
├── data/                                            # Processed MRI datasets and patient splits
├── data_downloads/                                  # Raw downloaded archives and tarballs
└── brats_env/                                       # Python virtual environment (PyTorch + CUDA)
```

---

## 3. Key Directories & Data Locations

### A. Model Source Code & Trainers (`src/`)
* **Custom Trainers:** Located in [`src/`](file:///mnt/DATA/EE23B009/EE23B009_BTP/src/). Includes classes like `nnUNetTrainer_Method3_InterLevel_SIAM_CurriculumBAA.py` (Flagship STAR-Net), `nnUNetTrainer_Decoder1_HalfRes.py`, and `nnUNetTrainer_M10_Sweeps.py`.
* **Loss Functions:** Located in [`src/losses/`](file:///mnt/DATA/EE23B009/EE23B009_BTP/src/losses/) (Boundary-Aware Attention loss, SIAM cross-scale hinge loss, Curriculum scheduling).
* **Evaluation & Testing:** Run any model on the 30-case held-out test split using `src/fast_evaluate_test30_all_sweeps.py` or `src/evaluate_saluja_test30.py`.

### B. Evaluation Metrics (`metrics_csv/`)
* Contains **25 CSV tables** recording per-patient and aggregated metrics (Mean/Median Dice, HD95, Sensitivity, Precision) across every tested architecture and hyperparameter setting.

### C. Trained Model Weights (`nnunet_data/nnUNet_results/`)
* Model checkpoints (`checkpoint_best.pth`, `checkpoint_final.pth`) are stored in the standard nnU-Net structure under:
  `/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results/`

### D. Presentation Assets (`results/presentation_figures/`)
* High-resolution visual figures used by [`build_visual_master_presentation.py`](file:///mnt/DATA/EE23B009/EE23B009_BTP/build_visual_master_presentation.py) to compile the PowerPoint deck. If modifying slides, edit this builder script and run it.

---

## 4. Environment Activation & nnU-Net Paths

To run evaluation, training, or preprocessing scripts, activate the existing virtual environment and configure the environment variables:

```bash
# 1. Activate the Python virtual environment
source /mnt/DATA/EE23B009/EE23B009_BTP/brats_env/bin/activate

# 2. Set nnU-Net path variables
export nnUNet_raw="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/mnt/DATA/EE23B009/EE23B009_BTP/nnunet_data/nnUNet_results"
```

---

## 5. Master Reference Summary

| Reference Document | Format | Description |
| :--- | :---: | :--- |
| [BTP_Phase1_Master_Presentation.pdf](file:///mnt/DATA/EE23B009/EE23B009_BTP/BTP_Phase1_Master_Presentation.pdf) | PDF / PPTX | **Main Entry Point**: 29 slides covering all concepts, architectures, and results. |
| [BTP_Final_Report.md](file:///mnt/DATA/EE23B009/EE23B009_BTP/BTP_Final_Report.md) | Markdown | Comprehensive technical documentation and mathematical formulations. |
| [BTP_IEEE_Report.pdf](file:///mnt/DATA/EE23B009/EE23B009_BTP/BTP_IEEE_Report.pdf) | PDF | 6-page IEEE-style conference manuscript. |
| [Literature_vs_Our_BTP_Results_Comparison.md](file:///mnt/DATA/EE23B009/EE23B009_BTP/Literature_vs_Our_BTP_Results_Comparison.md) | Markdown | Detailed benchmarking against literature (nnU-Net, MedNeXt, Saluja et al.). |
| [Literature_and_All_Configurations_Master_Benchmark.pdf](file:///mnt/DATA/EE23B009/EE23B009_BTP/Literature_and_All_Configurations_Master_Benchmark.pdf) | PDF | Consolidated tables comparing all 15 configurations. |

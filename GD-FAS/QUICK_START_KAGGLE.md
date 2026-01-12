# 🚀 Quick Start Guide - Using Kaggle Datasets with GD-FAS

This is a simplified guide to get you started quickly with your Kaggle datasets.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Your Kaggle datasets in `D:\CODIng\Machine Learning\FAS\datasets`

## Step-by-Step Setup

### 1️⃣ Install Dependencies

```bash
cd "D:\CODIng\Machine Learning\FAS\GD-FAS"
pip install -r requirement.txt
```

### 2️⃣ Prepare Your Datasets

Run the preparation script to extract frames and organize data:

```bash
python prepare_kaggle_dataset.py
```

**What this does:**
- ✅ Extracts every 5th frame from videos
- ✅ Organizes into train/test splits (80/20)
- ✅ Creates proper directory structure
- ✅ Categorizes into "attack" and "live" classes

**⏱️ Time estimate:** 15-60 minutes depending on data size

### 3️⃣ Verify Setup

Check if everything is ready:

```bash
python verify_dataset.py
```

This will show you:
- ✅ Dataset structure validation
- 📊 Statistics (number of videos, frames, class balance)
- ⚠️ Any issues or warnings

### 4️⃣ Train Your First Model

Start training with a simple protocol:

```bash
python training.py --protocol K_R_U_to_V --gs --log_name my_first_experiment
```

**What this does:**
- Trains on: 30k_fas (K), 98k_real (R), unidata_real (U)
- Tests on: archive (V)
- Enables group-wise scaling (--gs)
- Saves results to `results/my_first_experiment/`

## Dataset Codes Reference

| Code | Dataset | Type |
|------|---------|------|
| **K** | 30k_fas | Mixed (live + attacks) |
| **R** | 98k_real | Live only |
| **V** | archive | Mixed (live + various attacks) |
| **U** | unidata_real | Live only |

## More Training Examples

### Example 1: Train on attacks, test on real faces
```bash
python training.py --protocol K_V_to_R --gs --log_name attacks_to_real
```

### Example 2: Maximum training data
```bash
python training.py --protocol K_R_V_to_U --gs --log_name max_data --save
```
*Note: `--save` will save the best model checkpoint*

### Example 3: Quick test with single dataset
```bash
python training.py --protocol K_to_K --gs --log_name quick_test --max_iter 100
```

## Monitoring Training

Training will print:
- 📈 Loss per epoch
- 📊 Validation metrics: ACC, HTER, AUC, FPR@1%
- 🏆 Best HTER achieved

Example output:
```
epoch : 10 loss: 0.4523
--------------------------------------------
ACC_val:0.8534 HTER_val:0.1234 AUC:0.9234 fpr1p:0.0234 ECE:0.0456 acc:0.8765 threshold:0.5432
best_hter: 0.1234
--------------------------------------------
```

## Results Location

Find your results in:
```
results/
└── {log_name}/
    ├── {protocol}.txt        # Training log
    └── {protocol}_best.pth   # Best model (if --save used)
```

## Troubleshooting

### ❌ "No video folders found"
→ Run `python prepare_kaggle_dataset.py` first

### ❌ "CUDA out of memory"
→ Reduce batch size: `--batch_size 8` or `--batch_size 4`

### ❌ "ModuleNotFoundError"
→ Install dependencies: `pip install -r requirement.txt`

### ⚠️ Low accuracy
→ Try different protocols, increase `--max_iter`, or adjust learning rate

## Next Steps

1. ✅ Run preparation script
2. ✅ Verify datasets
3. ✅ Train your first model
4. 📊 Analyze results in `results/` directory
5. 🔬 Experiment with different protocols and parameters

## Need More Help?

- **Detailed guide**: See `KAGGLE_DATASET_SETUP.md`
- **Original paper**: See `README.md`
- **Code structure**: Explore `data/`, `models/`, `utils/` folders

## One-Command Setup (Windows)

For convenience, you can run:
```bash
prepare_and_train.bat
```

This will:
1. Install dependencies
2. Prepare datasets
3. Show you training options

---

**Happy Training! 🎉**

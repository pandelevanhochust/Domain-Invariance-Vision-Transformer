# 🎭 GD-FAS with Kaggle Datasets

> **Adapted for use with Kaggle Face Anti-Spoofing datasets**

This is an adapted version of the GD-FAS (ICCV 2025) implementation that works with your Kaggle datasets instead of the original research datasets.

## 📋 Table of Contents
- [Quick Start](#-quick-start)
- [Your Datasets](#-your-datasets)
- [Workflow](#-workflow)
- [Training Examples](#-training-examples)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Prepare Your Datasets
```bash
python prepare_kaggle_dataset.py
```
⏱️ This may take 15-60 minutes depending on your data size.

### 3. Verify Everything is Ready
```bash
python verify_dataset.py
```

### 4. Start Training
```bash
python training.py --protocol K_R_U_to_V --gs --log_name my_experiment
```

## 📊 Your Datasets

Your Kaggle datasets have been mapped to the GD-FAS protocol system:

| Code | Dataset Name | Location | Type | Videos/Samples |
|------|-------------|----------|------|----------------|
| **K** | 30k_fas | `datasets/30k_fas/` | Mixed | ~36 videos |
| **R** | 98k_real | `datasets/98k_real/` | Live only | ~30 folders |
| **V** | archive | `datasets/archive/` | Mixed | ~100+ samples |
| **U** | unidata_real | `datasets/unidata_real/` | Live only | 10 persons |

### Dataset Breakdown

#### 30k_fas (K)
- ✅ Live: videos and selfies
- ❌ Attacks: cut-out printouts, regular printouts, replay attacks

#### 98k_real (R)
- ✅ Live: real face videos and selfies
- ❌ Attacks: None (live-only dataset)

#### archive (V)
- ✅ Live: selfie images
- ❌ Attacks: 3D paper masks, cutout attacks, latex masks, silicone masks, replay attacks (display & mobile), textile masks, wrapped 3D masks

#### unidata_real (U)
- ✅ Live: real person videos
- ❌ Attacks: None (live-only dataset)

## 🔄 Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  Your Kaggle Datasets (Original)                           │
│  D:\CODIng\Machine Learning\FAS\datasets\                  │
│  ├── 30k_fas/          (videos + images)                   │
│  ├── 98k_real/         (videos + images)                   │
│  ├── archive/          (videos + images)                   │
│  └── unidata_real/     (videos)                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ python prepare_kaggle_dataset.py
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Processed Datasets (GD-FAS Format)                        │
│  D:\CODIng\Machine Learning\FAS\GD-FAS\datasets\           │
│  ├── 30k_fas/                                              │
│  │   ├── train/                                            │
│  │   │   ├── attack/  ← video_folders/  ← frames.png      │
│  │   │   └── live/    ← video_folders/  ← frames.png      │
│  │   └── test/                                             │
│  │       ├── attack/                                       │
│  │       └── live/                                         │
│  ├── 98k_real/  (same structure)                          │
│  ├── archive/   (same structure)                          │
│  └── unidata_real/  (same structure)                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ python verify_dataset.py
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Verification Report                                        │
│  ✓ Structure validated                                     │
│  ✓ Statistics displayed                                    │
│  ✓ Balance checked                                         │
│  ✓ Ready for training                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ python training.py --protocol ...
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Training & Results                                         │
│  results/{log_name}/                                        │
│  ├── {protocol}.txt        ← Training logs                 │
│  └── {protocol}_best.pth   ← Best model checkpoint         │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Training Examples

### Example 1: Cross-Domain Generalization (Recommended)
Train on multiple datasets, test on a different one:
```bash
python training.py --protocol K_R_U_to_V --gs --log_name cross_domain_1
```
- **Trains on:** 30k_fas, 98k_real, unidata_real
- **Tests on:** archive
- **Use case:** Test generalization to diverse attack types

### Example 2: Attack-Rich Training
```bash
python training.py --protocol K_V_to_R --gs --log_name attack_rich --save
```
- **Trains on:** 30k_fas, archive (both have attacks)
- **Tests on:** 98k_real (live faces)
- **Use case:** Test if model can correctly identify real faces after seeing many attacks

### Example 3: Maximum Data
```bash
python training.py --protocol K_R_V_to_U --gs --log_name max_data --save
```
- **Trains on:** 30k_fas, 98k_real, archive
- **Tests on:** unidata_real
- **Use case:** Use all available data for training

### Example 4: Quick Test (Fast)
```bash
python training.py --protocol K_to_K --gs --log_name quick_test --max_iter 100 --batch_size 8
```
- **Trains on:** 30k_fas
- **Tests on:** 30k_fas
- **Use case:** Quick pipeline test (not for real evaluation)

### Example 5: Custom Parameters
```bash
python training.py \
  --protocol K_R_U_to_V \
  --gs \
  --log_name custom_experiment \
  --backbone clip \
  --batch_size 16 \
  --max_iter 400 \
  --lr 0.000003 \
  --temperature 0.1 \
  --save \
  --seed 2025
```

## 📈 Results

### Understanding the Metrics

During training, you'll see metrics like:
```
ACC_val:0.8534 HTER_val:0.1234 AUC:0.9234 fpr1p:0.0234 ECE:0.0456 acc:0.8765 threshold:0.5432
```

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **HTER** | Half Total Error Rate | Lower is better (< 0.10 is good) |
| **AUC** | Area Under ROC Curve | Higher is better (> 0.95 is good) |
| **ACC** | Accuracy | Higher is better (> 0.90 is good) |
| **fpr1p** | False Positive Rate at 1% | Lower is better |
| **ECE** | Expected Calibration Error | Lower is better |

### Where to Find Results

```
results/
└── {your_log_name}/
    ├── {protocol}.txt        # Full training log
    └── {protocol}_best.pth   # Best model (if --save used)
```

Example log file content:
```
------------------------------------------------------
information
------------------------------------------------------
log name             : my_experiment
protocol name        : K_R_U_to_V
backbone             : clip
batch size           : 16
...
------------------------------------------------------
training
------------------------------------------------------
epoch : 10 loss: 0.4523
--------------------------------------------
ACC_val:0.8534 HTER_val:0.1234 AUC:0.9234 ...
best_hter: 0.1234
--------------------------------------------
...
```

## 🔧 Troubleshooting

### Problem: "No video folders found"
**Solution:**
```bash
python prepare_kaggle_dataset.py
```
Make sure it completes successfully.

### Problem: "CUDA out of memory"
**Solution:** Reduce batch size
```bash
python training.py --protocol K_to_V --gs --batch_size 4
```

### Problem: "ModuleNotFoundError: No module named 'cv2'"
**Solution:** Install OpenCV
```bash
pip install opencv-python
```

### Problem: Slow training
**Solutions:**
- Use smaller backbone: `--backbone resnet18`
- Reduce iterations: `--max_iter 200`
- Check GPU is being used: `torch.cuda.is_available()` should return `True`

### Problem: Low accuracy/high HTER
**Solutions:**
- Try different protocols (different train/test splits)
- Increase training iterations: `--max_iter 600`
- Adjust learning rate: `--lr 0.00001`
- Check dataset balance with `verify_dataset.py`

### Problem: Imbalanced dataset warning
**Solution:** This is expected for R and U (live-only datasets). Use them as test sets or combine with attack datasets (K or V) for training.

## 📚 Additional Resources

- **Quick Start Guide:** `QUICK_START_KAGGLE.md` - Minimal steps to get started
- **Detailed Setup:** `KAGGLE_DATASET_SETUP.md` - Comprehensive guide
- **Changes Made:** `CHANGES_SUMMARY.md` - What was modified for Kaggle datasets
- **Original Paper:** `README.md` - Original GD-FAS paper and methodology

## 🎓 Original Paper

This implementation is based on:
> **Group-wise Scaling and Orthogonal Decomposition for Domain-Invariant Feature Extraction in Face Anti-Spoofing** (ICCV 2025)
> 
> Seungjin Jung, Kanghee Lee, Younghyung Jeong, Haeun Noh, Jungmin Lee, and Jongwon Choi

[[Arxiv](https://arxiv.org/abs/2507.04006)] [[Project](https://seungjinjung.github.io/project/GD-FAS.html)]

## 🤝 Protocol Naming Convention

The protocol format is: `TRAIN_DATASETS_to_TEST_DATASETS`

**Available codes:**
- **K** = 30k_fas
- **R** = 98k_real  
- **V** = archive
- **U** = unidata_real

**Examples:**
- `K_to_V` → Train on K, test on V
- `K_R_to_U` → Train on K and R, test on U
- `K_R_V_to_U` → Train on K, R, and V, test on U

## 💻 System Requirements

- **OS:** Windows 10/11, Linux, or macOS
- **Python:** 3.8 or higher
- **GPU:** CUDA-capable GPU recommended (NVIDIA)
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** ~10GB for processed datasets (depends on your data)

## ⚡ Performance Tips

1. **Use GPU:** Training on CPU is very slow
2. **Batch Size:** Adjust based on your GPU memory
3. **Num Workers:** Increase in `data/__init__.py` if you have many CPU cores
4. **Mixed Precision:** Consider adding AMP for faster training (advanced)

## 🎉 You're Ready!

Follow the Quick Start section above and you'll be training in minutes!

For any issues, check the troubleshooting section or the detailed guides.

---

**Happy Training! 🚀**

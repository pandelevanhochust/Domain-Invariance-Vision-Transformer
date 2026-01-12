# 🎯 START HERE - GD-FAS with Your Kaggle Datasets

Welcome! This guide will get you up and running quickly.

## 🎬 Three Ways to Get Started

### Option 1: Interactive Wizard (Easiest) ⭐
Run the interactive setup wizard that guides you through everything:
```bash
python setup_wizard.py
```
This will:
- ✅ Check your system
- ✅ Install dependencies
- ✅ Prepare datasets
- ✅ Verify setup
- ✅ Help you start training

**Best for:** First-time users who want guidance

---

### Option 2: Quick Commands (Fast)
If you prefer to run commands directly:

```bash
# 1. Install dependencies
pip install -r requirement.txt

# 2. Prepare datasets (15-60 minutes)
python prepare_kaggle_dataset.py

# 3. Verify setup
python verify_dataset.py

# 4. Start training
python training.py --protocol K_R_U_to_V --gs --log_name my_experiment
```

**Best for:** Users comfortable with command line

---

### Option 3: Windows Batch File (One-Click)
Double-click on:
```
prepare_and_train.bat
```
This will install dependencies and prepare datasets automatically.

**Best for:** Windows users who prefer GUI

---

## 📚 Documentation Guide

We've created several guides for different needs:

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **START_HERE.md** | This file - quick overview | First time here |
| **QUICK_START_KAGGLE.md** | Minimal steps to get started | Want to start fast |
| **README_KAGGLE.md** | Complete overview with examples | Want full picture |
| **KAGGLE_DATASET_SETUP.md** | Detailed setup instructions | Need detailed help |
| **CHANGES_SUMMARY.md** | What was modified | Curious about changes |
| **README.md** | Original paper information | Academic reference |

---

## 🗺️ Your Datasets

Your Kaggle datasets will be used like this:

```
Original Location:                    After Processing:
D:\CODIng\Machine Learning\          D:\CODIng\Machine Learning\
FAS\datasets\                        FAS\GD-FAS\datasets\
├── 30k_fas/                         ├── 30k_fas/
├── 98k_real/                        │   ├── train/
├── archive/                         │   │   ├── attack/
└── unidata_real/                    │   │   └── live/
                                     │   └── test/
    ↓ prepare_kaggle_dataset.py      │       ├── attack/
                                     │       └── live/
                                     ├── 98k_real/
                                     ├── archive/
                                     └── unidata_real/
```

**Dataset Codes:**
- **K** = 30k_fas (mixed: live + attacks)
- **R** = 98k_real (live only)
- **V** = archive (mixed: live + various attacks)
- **U** = unidata_real (live only)

---

## 🚀 Quick Training Examples

Once setup is complete, try these:

### Beginner: Quick Test
```bash
python training.py --protocol K_to_K --gs --log_name test --max_iter 100
```
Fast test to ensure everything works (~10 minutes)

### Intermediate: Cross-Domain
```bash
python training.py --protocol K_R_U_to_V --gs --log_name exp1
```
Train on 3 datasets, test on 1 (~2-4 hours)

### Advanced: Full Training with Save
```bash
python training.py --protocol K_R_V_to_U --gs --log_name exp2 --save --max_iter 600
```
Maximum data, save best model (~4-6 hours)

---

## 📊 Understanding Results

Results are saved in `results/{log_name}/`:

```
results/
└── my_experiment/
    ├── K_R_U_to_V.txt        # Training log
    └── K_R_U_to_V_best.pth   # Best model (if --save used)
```

**Key Metrics:**
- **HTER** (Half Total Error Rate): Lower is better, < 0.10 is good
- **AUC** (Area Under Curve): Higher is better, > 0.95 is good
- **ACC** (Accuracy): Higher is better, > 0.90 is good

---

## ❓ Common Questions

### Q: How long does setup take?
**A:** 
- Installing dependencies: 5-10 minutes
- Preparing datasets: 15-60 minutes (depends on data size)
- First training run: 2-4 hours

### Q: Do I need a GPU?
**A:** Highly recommended. Training on CPU is very slow (10-20x slower).

### Q: What if I get "CUDA out of memory"?
**A:** Reduce batch size: `--batch_size 8` or `--batch_size 4`

### Q: Can I use only some of my datasets?
**A:** Yes! Just use the codes you want. E.g., `K_to_V` uses only 30k_fas and archive.

### Q: How do I know if it's working?
**A:** Run `python verify_dataset.py` to check everything is ready.

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No video folders found" | Run `python prepare_kaggle_dataset.py` |
| "CUDA out of memory" | Use `--batch_size 4` |
| "ModuleNotFoundError" | Run `pip install -r requirement.txt` |
| Slow training | Check GPU is available, reduce `--max_iter` |
| Low accuracy | Try different protocols, increase iterations |

---

## 📋 Checklist

Before training, make sure:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirement.txt`)
- [ ] Datasets prepared (`python prepare_kaggle_dataset.py`)
- [ ] Setup verified (`python verify_dataset.py`)
- [ ] GPU available (optional but recommended)

---

## 🎓 What is GD-FAS?

GD-FAS (Group-wise Scaling and Orthogonal Decomposition for Face Anti-Spoofing) is a research paper from ICCV 2025 that achieves state-of-the-art results in detecting face spoofing attacks.

**Key Features:**
- Domain generalization (works across different datasets)
- Group-wise scaling (the `--gs` flag)
- CLIP-based backbone for better features

**Original Paper:** [Arxiv Link](https://arxiv.org/abs/2507.04006)

---

## 🎯 Next Steps

1. **Choose your path:** Interactive wizard, quick commands, or batch file
2. **Follow the steps:** Install → Prepare → Verify → Train
3. **Check results:** Look in `results/` folder
4. **Experiment:** Try different protocols and parameters

---

## 💡 Pro Tips

1. **Start small:** Use `K_to_K` with `--max_iter 100` to test the pipeline
2. **Save models:** Add `--save` to keep the best checkpoint
3. **Monitor training:** Watch the HTER metric - lower is better
4. **Experiment:** Try different train/test combinations
5. **Read logs:** Check `results/{log_name}/{protocol}.txt` for details

---

## 📞 Need More Help?

- **Quick reference:** See `QUICK_START_KAGGLE.md`
- **Detailed guide:** See `KAGGLE_DATASET_SETUP.md`
- **Full overview:** See `README_KAGGLE.md`
- **Technical details:** See `CHANGES_SUMMARY.md`

---

## ✨ Ready to Start?

Choose your preferred method above and begin!

**Recommended for first-time users:**
```bash
python setup_wizard.py
```

**Good luck! 🚀**

---

<div align="center">

**Made with ❤️ for Face Anti-Spoofing Research**

[Original Paper](https://arxiv.org/abs/2507.04006) | [Project Page](https://seungjinjung.github.io/project/GD-FAS.html)

</div>

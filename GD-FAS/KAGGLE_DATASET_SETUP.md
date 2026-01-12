# Kaggle Dataset Setup Guide for GD-FAS

This guide explains how to use your Kaggle datasets with the GD-FAS framework.

## Overview

Your available Kaggle datasets have been mapped to the GD-FAS protocol system:

| Code | Dataset Name | Description |
|------|-------------|-------------|
| K | 30k_fas | Contains live videos/selfies and various attack types (printouts, replay, cut-out) |
| R | 98k_real | Real/live face samples |
| V | archive | Various attack types (3D masks, cutouts, replay, silicone masks, etc.) and selfies |
| U | unidata_real | Real/live face samples from different persons |

## Step 1: Prepare the Datasets

Run the preparation script to extract frames from videos and organize the data:

```bash
python prepare_kaggle_dataset.py
```

This script will:
- Extract every 5th frame from all videos (as per SAFAS methodology)
- Organize images into the required structure:
  ```
  datasets/
  ├── 30k_fas/
  │   ├── train/
  │   │   ├── attack/
  │   │   └── live/
  │   └── test/
  │       ├── attack/
  │       └── live/
  ├── 98k_real/
  ├── archive/
  └── unidata_real/
  ```
- Split data into 80% training and 20% testing
- Display statistics about the processed datasets

**Note**: This process may take a while depending on the number of videos!

## Step 2: Training the Model

After preparing the datasets, you can train the model using various protocols.

### Example Training Commands

#### Using all Kaggle datasets (cross-domain generalization)
Train on three datasets, test on one:
```bash
python training.py --protocol K_R_U_to_V --gs --log_name kaggle_experiment1
```
This trains on 30k_fas, 98k_real, and unidata_real, then tests on archive.

#### Mixed protocol example
```bash
python training.py --protocol K_V_to_R --gs --log_name kaggle_experiment2
```
This trains on 30k_fas and archive, then tests on 98k_real.

#### Single dataset training and testing
```bash
python training.py --protocol K_to_K --gs --log_name single_dataset_test
```

### Protocol Format

The protocol format is: `TRAIN_DATASETS_to_TEST_DATASETS`

Where datasets are separated by underscores. Examples:
- `K_to_V` - Train on 30k_fas, test on archive
- `K_R_to_U` - Train on 30k_fas and 98k_real, test on unidata_real
- `K_V_U_to_R` - Train on 30k_fas, archive, and unidata_real, test on 98k_real

### Important Training Parameters

- `--gs` : Enable group-wise scaling (recommended, this is the paper's main contribution)
- `--protocol` : Specify training and testing datasets
- `--log_name` : Name for your experiment (results will be saved in `results/{log_name}/`)
- `--backbone` : Choose backbone architecture (`clip` or `resnet18`, default is `clip`)
- `--batch_size` : Batch size (default: 16)
- `--max_iter` : Maximum iterations (default: 400)
- `--lr` : Learning rate (default: 0.000003)
- `--save` : Save the best model checkpoint

### Full Training Example with All Parameters

```bash
python training.py \
  --protocol K_R_U_to_V \
  --gs \
  --log_name my_experiment \
  --backbone clip \
  --batch_size 16 \
  --max_iter 400 \
  --lr 0.000003 \
  --save \
  --seed 2025
```

## Step 3: Check Results

Results will be saved in `results/{log_name}/`:
- Training logs: `results/{log_name}/{protocol}.txt`
- Best model (if `--save` is used): `results/{log_name}/{protocol}_best.pth`

The log file contains:
- Training loss per epoch
- Validation metrics: ACC, HTER, AUC, FPR@1%, ECE, accuracy, threshold
- Best HTER achieved

## Dataset Recommendations

### For Cross-Domain Generalization Testing

1. **Diverse Attack Types**: Use archive (V) as test set since it contains many attack types
   ```bash
   python training.py --protocol K_R_U_to_V --gs --log_name diverse_attacks
   ```

2. **Real Face Generalization**: Use 98k_real (R) as test set
   ```bash
   python training.py --protocol K_V_U_to_R --gs --log_name real_faces
   ```

3. **Maximum Training Data**: Train on K, R, V together
   ```bash
   python training.py --protocol K_R_V_to_U --gs --log_name maximum_data
   ```

### For Quick Testing

If you want to test the pipeline quickly with less data, you can modify the preparation script to process only a subset of your datasets or reduce the number of videos per category.

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (try 8 or 4)
- Ensure you have enough GPU memory

### No videos/frames found
- Make sure the preparation script completed successfully
- Check that the `datasets` folder exists in the GD-FAS directory
- Verify the folder structure matches the expected format

### Import errors
- Install all requirements: `pip install -r requirement.txt`
- Make sure OpenCV and tqdm are installed

### Slow training
- Enable CUDA if available
- Increase `--num_workers` in the DataLoader (edit `data/__init__.py`)
- Use a smaller backbone (`--backbone resnet18` instead of `clip`)

## Dataset Statistics

After running the preparation script, you'll see statistics like:

```
30k_fas:
  train/attack: 28 videos, 5234 frames
  train/live: 21 videos, 3892 frames
  test/attack: 8 videos, 1453 frames
  test/live: 5 videos, 982 frames
```

This helps you understand the data distribution and balance.

## Next Steps

1. Run `prepare_kaggle_dataset.py` to process your data
2. Choose a protocol based on your experimental needs
3. Train the model with `training.py`
4. Evaluate results in the `results/` directory
5. Experiment with different protocols and hyperparameters

For the original paper's methodology, refer to the main README.md.

## Combining with Original Datasets (Optional)

If you later obtain the original datasets (CASIA, OULU, MSU, etc.), you can:
1. Process them according to the main README.md instructions
2. Place them in the `datasets` folder alongside your Kaggle datasets
3. Use mixed protocols, e.g., `O_K_to_V` (train on OULU and 30k_fas, test on archive)

The protocol decoder supports both original and Kaggle datasets simultaneously!

@echo off
echo ============================================================
echo GD-FAS Kaggle Dataset Setup and Training
echo ============================================================
echo.

echo Step 1: Installing required packages...
pip install -r requirement.txt
echo.

echo Step 2: Preparing datasets (this may take a while)...
python prepare_kaggle_dataset.py
echo.

echo ============================================================
echo Dataset preparation complete!
echo ============================================================
echo.
echo You can now train the model with commands like:
echo   python training.py --protocol K_R_U_to_V --gs --log_name my_experiment
echo.
echo Available dataset codes:
echo   K = 30k_fas
echo   R = 98k_real
echo   V = archive
echo   U = unidata_real
echo.
echo Example protocols:
echo   K_R_U_to_V  (train on K,R,U; test on V)
echo   K_V_to_R    (train on K,V; test on R)
echo   K_to_K      (single dataset train/test)
echo.
echo For more information, see KAGGLE_DATASET_SETUP.md
echo ============================================================
python training.py --gs --protocol O_C_M_to_I

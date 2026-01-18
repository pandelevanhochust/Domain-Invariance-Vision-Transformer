

python training.py --backbone clip --gs --temperature 0.1 --protocol C_FAS_to_C_FAS --save --log_name MyCustomFAS_Run --data_root dataset


python training.py --backbone clip --gs --temperature 0.1 --protocol C_FAS_to_C_FAS --data_root dataset/CrossVal/Fold1 --log_name Run_Fold1

python training.py --backbone clip --gs --temperature 0.1 --data_root dataset/CrossVal/Fold2 --log_name Run_Fold2

python training.py --backbone clip --gs --temperature 0.1 --data_root dataset/CrossVal/Fold3 --log_name Run_Fold3


# Original dataset protocols (if you have OULU, CASIA, Idiap, MSU datasets)
# python training.py --gs --temperature 0.1 --protocol O_C_I_to_M
# python training.py --gs --temperature 0.1 --protocol O_M_I_to_C
# python training.py --gs --temperature 1.1 --protocol O_C_M_to_I
# python training.py --gs --temperature 1.0 --protocol I_C_M_to_O

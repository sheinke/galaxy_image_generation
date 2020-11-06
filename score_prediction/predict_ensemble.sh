python judge.py --model_file ./ensemble_models/model_1_unfrozen.h5 --score_pred_mean 1.6517098281304168 --score_pred_std 1.0836055985878632 --out_csv model_1_scores.csv   --img_dir ./cil_data/cosmology_aux_data_170429/cosmology_aux_data_170429/query
python judge.py --model_file ./ensemble_models/model_2_unfrozen.h5 --score_pred_mean 1.6525513001952545 --score_pred_std 1.0817796577706593 --out_csv model_2_scores.csv   --img_dir ./cil_data/cosmology_aux_data_170429/cosmology_aux_data_170429/query
python judge.py --model_file ./ensemble_models/model_3_unfrozen.h5 --score_pred_mean 1.6537725628881483 --score_pred_std 1.0874072678932671 --out_csv model_3_scores.csv   --img_dir ./cil_data/cosmology_aux_data_170429/cosmology_aux_data_170429/query
python judge.py --model_file ./ensemble_models/model_4_unfrozen.h5 --score_pred_mean 1.658948185127338  --score_pred_std 1.0848215225678446 --out_csv model_4_scores.csv   --img_dir ./cil_data/cosmology_aux_data_170429/cosmology_aux_data_170429/query
python judge.py --model_file ./ensemble_models/model_5_unfrozen.h5 --score_pred_mean 1.652401341553912  --score_pred_std 1.0803430161731886 --out_csv model_5_scores.csv   --img_dir ./cil_data/cosmology_aux_data_170429/cosmology_aux_data_170429/query
python average_predictions.py
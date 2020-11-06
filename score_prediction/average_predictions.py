import pandas as pd

pred_1 = pd.read_csv("model_1_scores.csv").set_index("Id")
pred_2 = pd.read_csv("model_2_scores.csv").set_index("Id")
pred_3 = pd.read_csv("model_3_scores.csv").set_index("Id")
pred_4 = pd.read_csv("model_4_scores.csv").set_index("Id")
pred_5 = pd.read_csv("model_5_scores.csv").set_index("Id")

pred_final = pred_1.add(pred_2).add(pred_3).add(pred_4).add(pred_5)
pred_final["Predicted"] = pred_final["Predicted"] / 5.

pred_final.reset_index().to_csv("ensemble_predictions.csv", index=False)
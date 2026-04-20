

# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np

# # =========================
# # CONFIG
# # =========================

# BUNDLE_PATH = r"C:\Users\tavis\OneDrive\Documents\BARC\work\fk_full_bundle.pth"

# TEST_FILE = r"C:\Users\tavis\OneDrive\Documents\BARC\work\newtest_remaining.xlsx"

# OUTPUT_FILE = r"C:\Users\tavis\OneDrive\Documents\BARC\work\newfk_predictions.xlsx"

# INPUT_COLS  = ["Leg1","Leg2","Leg3","Leg4","Leg5","Leg6"]
# OUTPUT_COLS = ["X","Y","Z","Thetax","Thetay","Thetaz"]

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# print("Using device:", DEVICE)

# # =========================
# # MODEL ARCHITECTURE
# # =========================

# class FKNet(nn.Module):

#     def __init__(self):

#         super().__init__()

#         self.net = nn.Sequential(

#             nn.Linear(6,128),
#             nn.LayerNorm(128),
#             nn.GELU(),

#             nn.Linear(128,256),
#             nn.LayerNorm(256),
#             nn.GELU(),

#             nn.Linear(256,256),
#             nn.LayerNorm(256),
#             nn.GELU(),

#             nn.Linear(256,128),
#             nn.LayerNorm(128),
#             nn.GELU(),

#             nn.Linear(128,64),
#             nn.GELU(),

#             nn.Linear(64,6)

#         )

#     def forward(self,x):
#         return self.net(x)

# # =========================
# # LOAD BUNDLE FILE
# # =========================

# print("Loading bundle file...")

# checkpoint = torch.load(BUNDLE_PATH, map_location=DEVICE)

# model = FKNet().to(DEVICE)

# model.load_state_dict(checkpoint["model_state_dict"])

# model.eval()

# x_mean = checkpoint["x_mean"]
# x_std  = checkpoint["x_std"]

# y_mean = checkpoint["y_mean"]
# y_std  = checkpoint["y_std"]

# print("Bundle loaded successfully")

# # =========================
# # LOAD TEST DATA
# # =========================

# print("Loading test dataset...")

# test_df = pd.read_excel(TEST_FILE)

# Xt = test_df[INPUT_COLS].values.astype(np.float32)
# Yt = test_df[OUTPUT_COLS].values.astype(np.float32)

# # =========================
# # NORMALIZE INPUT
# # =========================

# Xt_n = (Xt - x_mean) / x_std

# Xt_n = torch.tensor(Xt_n, dtype=torch.float32).to(DEVICE)

# # =========================
# # MODEL PREDICTION
# # =========================

# print("Running FK predictions...")

# with torch.no_grad():

#     pred_n = model(Xt_n).cpu().numpy()

# pred = pred_n * y_std + y_mean

# # =========================
# # ERROR CALCULATION
# # =========================

# trans_err = np.linalg.norm(pred[:,0:3] - Yt[:,0:3], axis=1)

# rot_err = np.linalg.norm(pred[:,3:6] - Yt[:,3:6], axis=1)

# mean_trans_err = np.mean(trans_err)

# mean_rot_err = np.mean(rot_err)

# max_trans_err = np.max(trans_err)

# max_rot_err = np.max(rot_err)

# # threshold statistics

# trans_above_01 = np.sum(trans_err > 0.1)

# rot_above_01 = np.sum(rot_err > 0.1)

# trans_within_01_percent = np.mean(trans_err < 0.1) * 100

# rot_within_01_percent = np.mean(rot_err < 0.1) * 100

# print("\n===== Accuracy Report =====")

# print("Mean translation error (mm):", mean_trans_err)

# print("Mean rotation error:", mean_rot_err)

# print("Max translation error:", max_trans_err)

# print("Max rotation error:", max_rot_err)

# print("\nPoints with translation error > 0.1 mm:", trans_above_01)

# print("Points with rotation error > 0.1:", rot_above_01)

# print("\n% points within 0.1 mm translation:", trans_within_01_percent)

# print("% points within 0.1 rotation:", rot_within_01_percent)

# # =========================
# # SAVE RESULTS
# # =========================

# results_df = pd.DataFrame()

# for col in INPUT_COLS:
#     results_df[col] = test_df[col]

# for i,col in enumerate(OUTPUT_COLS):
#     results_df["True_"+col] = Yt[:,i]

# for i,col in enumerate(OUTPUT_COLS):
#     results_df["Pred_"+col] = pred[:,i]

# results_df["Translation_Error_mm"] = trans_err
# results_df["Rotation_Error"] = rot_err

# # summary sheet

# summary_df = pd.DataFrame({

#     "Metric":[
#         "Mean Translation Error (mm)",
#         "Mean Rotation Error",
#         "Max Translation Error",
#         "Max Rotation Error",
#         "Points with Translation Error > 0.1 mm",
#         "Points with Rotation Error > 0.1",
#         "% within 0.1 mm Translation",
#         "% within 0.1 Rotation"
#     ],

#     "Value":[
#         mean_trans_err,
#         mean_rot_err,
#         max_trans_err,
#         max_rot_err,
#         trans_above_01,
#         rot_above_01,
#         trans_within_01_percent,
#         rot_within_01_percent
#     ]

# })

# print("Saving results...")

# with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

#     results_df.to_excel(writer,
#                         sheet_name="Predictions",
#                         index=False)

#     summary_df.to_excel(writer,
#                         sheet_name="Summary",
#                         index=False)

# print("Saved to:", OUTPUT_FILE)

# # =========================
# # MANUAL INPUT MODE
# # =========================

# def manual_fk_predict():

#     print("\n===== Manual FK Prediction Mode =====")

#     print("Enter 6 leg lengths")

#     print("Example: 450 455 448 460 452 451")

#     print("Type 'exit' to quit\n")

#     while True:

#         user_input = input("Leg lengths > ")

#         if user_input.lower().strip() == "exit":
#             break

#         try:

#             legs = list(map(float,
#                             user_input.strip().split()))

#             if len(legs) != 6:
#                 print("Enter exactly 6 values")
#                 continue

#             X = np.array(legs,
#                          dtype=np.float32).reshape(1,-1)

#             Xn = (X - x_mean) / x_std

#             Xn_t = torch.tensor(Xn,
#                                 dtype=torch.float32).to(DEVICE)

#             with torch.no_grad():

#                 pred_n = model(Xn_t).cpu().numpy()

#             pred = pred_n * y_std + y_mean

#             pred = pred[0]

#             # print("\nPredicted Pose")

#             # print("X      :", pred[0])
#             # print("Y      :", pred[1])
#             # print("Z      :", pred[2])
#             # print("Thetax :", pred[3])
#             # print("Thetay :", pred[4])
#             # print("Thetaz :", pred[5])
#             print(f"{pred[0]},{pred[1]},{pred[2]},{pred[3]},{pred[4]},{pred[5]}")

#             print()

#         except Exception as e:

#             print("Invalid input:", e)

# # =========================
# # RUN MANUAL MODE
# # =========================

# # manual_fk_predict()
# if __name__ == "__main__":
#     manual_fk_predict()


# ###############################################################################################################################################

# ##############################################################################################################################################



import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys

# =========================
# CONFIG (same folder paths)
# =========================

BUNDLE_PATH = "fk_full_bundle.pth"
TEST_FILE = "newtest_remaining.xlsx"
OUTPUT_FILE = "newfk_predictions.xlsx"

INPUT_COLS  = ["Leg1","Leg2","Leg3","Leg4","Leg5","Leg6"]
OUTPUT_COLS = ["X","Y","Z","Thetax","Thetay","Thetaz"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# MODEL
# =========================

class FKNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(6,128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128,256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256,256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256,128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128,64),
            nn.GELU(),

            nn.Linear(64,6)
        )

    def forward(self,x):
        return self.net(x)

# =========================
# LOAD MODEL
# =========================

checkpoint = torch.load(BUNDLE_PATH, map_location=DEVICE)

model = FKNet().to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

x_mean = checkpoint["x_mean"]
x_std  = checkpoint["x_std"]

y_mean = checkpoint["y_mean"]
y_std  = checkpoint["y_std"]

# =========================
# EXCEL MODE
# =========================

def run_excel_pipeline():

    test_df = pd.read_excel(TEST_FILE)

    Xt = test_df[INPUT_COLS].values.astype(np.float32)
    Yt = test_df[OUTPUT_COLS].values.astype(np.float32)

    Xt_n = (Xt - x_mean) / x_std
    Xt_n = torch.tensor(Xt_n, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred_n = model(Xt_n).cpu().numpy()

    pred = pred_n * y_std + y_mean

    trans_err = np.linalg.norm(pred[:,0:3] - Yt[:,0:3], axis=1)
    rot_err   = np.linalg.norm(pred[:,3:6] - Yt[:,3:6], axis=1)

    results_df = pd.DataFrame()

    for col in INPUT_COLS:
        results_df[col] = test_df[col]

    for i,col in enumerate(OUTPUT_COLS):
        results_df["True_"+col] = Yt[:,i]

    for i,col in enumerate(OUTPUT_COLS):
        results_df["Pred_"+col] = pred[:,i]

    results_df["Translation_Error_mm"] = trans_err
    results_df["Rotation_Error"] = rot_err

    results_df.to_excel(OUTPUT_FILE,index=False)

    print("Excel prediction complete")

# =========================
# MANUAL MODE
# =========================

def manual_fk_predict():

    user_input = sys.stdin.read().strip()

    legs = list(map(float,user_input.split()))

    X = np.array(legs,
                 dtype=np.float32).reshape(1,-1)

    Xn = (X - x_mean) / x_std

    Xn_t = torch.tensor(Xn,
                        dtype=torch.float32).to(DEVICE)

    with torch.no_grad():

        pred_n = model(Xn_t).cpu().numpy()

    pred = pred_n * y_std + y_mean

    pred = pred[0]

    print(f"{pred[0]},{pred[1]},{pred[2]},{pred[3]},{pred[4]},{pred[5]}")

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "excel":

        run_excel_pipeline()

    elif len(sys.argv) > 1 and sys.argv[1] == "manual":

        manual_fk_predict()

    else:

        print("Usage:")
        print("python fk_predict.py excel")
        print("python fk_predict.py manual")
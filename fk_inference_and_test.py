
import torch
import torch.nn as nn
import numpy as np
import sys

# =========================
# CONFIG
# =========================

BUNDLE_PATH = "fk_full_bundle.pth"
INPUT_FILE  = "input.txt"
OUTPUT_FILE = "output.txt"
CHECKSUM_FILE = "checksum_output.txt"

MIN_LEG = 168.0
MAX_LEG = 240.0

CHECKSUM_ENABLED = False   # 🔁 TOGGLE HERE

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
# VALIDATION
# =========================

def is_valid_legs(legs):
    return len(legs) == 6 and all(MIN_LEG <= l <= MAX_LEG for l in legs)

# =========================
# CHECKSUM FUNCTION
# =========================

def compute_difference(prev, curr):

    trans_diff = np.linalg.norm(curr[0:3] - prev[0:3])
    rot_diff   = np.linalg.norm(curr[3:6] - prev[3:6])

    return trans_diff, rot_diff

# =========================
# FILE STREAM MODE
# =========================

def run_txt_pipeline():

    print("\n📡 Streaming input processing...\n")

    prev_pose = None

    out_file = CHECKSUM_FILE if CHECKSUM_ENABLED else OUTPUT_FILE

    with open(INPUT_FILE, "r") as f_in, open(out_file, "a") as f_out:

        for line in f_in:

            if not line.strip():
                continue

            try:
                legs = list(map(float, line.strip().split()))
            except:
                f_out.write("INVALID\n")
                continue

            if not is_valid_legs(legs):
                f_out.write("INVALID\n")
                continue

            X = np.array(legs, dtype=np.float32).reshape(1,-1)
            Xn = (X - x_mean) / x_std
            Xn_t = torch.tensor(Xn, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                pred_n = model(Xn_t).cpu().numpy()

            pred = (pred_n * y_std + y_mean)[0]

            # ================= CHECKSUM =================
            if CHECKSUM_ENABLED and prev_pose is not None:

                trans_diff, rot_diff = compute_difference(prev_pose, pred)

                f_out.write(
                    f"POSE {' '.join(map(str,pred))} | "
                    f"dT={trans_diff:.4f} dR={rot_diff:.4f}\n"
                )

            else:
                f_out.write(" ".join(map(str, pred)) + "\n")

            prev_pose = pred

    print(f"✅ Output saved to {out_file}\n")

# =========================
# MANUAL MODE
# =========================

def manual_interactive():

    print(f"\nEnter 6 leg values ({MIN_LEG} to {MAX_LEG}) or type 'exit'")

    prev_pose = None

    while True:

        user_input = input("Legs > ")

        if user_input.lower() == "exit":
            break

        try:
            legs = list(map(float, user_input.split()))
        except:
            print("Invalid input")
            continue

        if not is_valid_legs(legs):
            print("❌ Out of range")
            continue

        X = np.array(legs, dtype=np.float32).reshape(1,-1)
        Xn = (X - x_mean) / x_std
        Xn_t = torch.tensor(Xn, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred_n = model(Xn_t).cpu().numpy()

        pred = (pred_n * y_std + y_mean)[0]

        print("Pred:", pred)

        out_file = CHECKSUM_FILE if CHECKSUM_ENABLED else OUTPUT_FILE

        with open(out_file, "a") as f:

            if CHECKSUM_ENABLED and prev_pose is not None:

                trans_diff, rot_diff = compute_difference(prev_pose, pred)

                f.write(
                    f"MANUAL {' '.join(map(str,pred))} | "
                    f"dT={trans_diff:.4f} dR={rot_diff:.4f}\n"
                )

            else:
                f.write(" ".join(map(str, pred)) + "\n")

        prev_pose = pred

# =========================
# C++ MODE
# =========================

def manual_stdin():

    user_input = sys.stdin.read().strip()

    try:
        legs = list(map(float, user_input.split()))
    except:
        print("ERROR")
        return

    if not is_valid_legs(legs):
        print("ERROR")
        return

    X = np.array(legs, dtype=np.float32).reshape(1,-1)
    Xn = (X - x_mean) / x_std
    Xn_t = torch.tensor(Xn, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred_n = model(Xn_t).cpu().numpy()

    pred = (pred_n * y_std + y_mean)[0]

    print(" ".join(map(str, pred)))

# =========================
# MENU
# =========================

def menu():

    while True:

        print("\n===== FK Prediction Menu =====")
        print("1. Stream from input.txt")
        print("2. Manual input")
        print("3. Toggle Checksum")
        print("4. Exit")

        choice = input("Select option: ")

        global CHECKSUM_ENABLED

        if choice == "1":
            run_txt_pipeline()

        elif choice == "2":
            manual_interactive()

        elif choice == "3":
            CHECKSUM_ENABLED = not CHECKSUM_ENABLED
            print(f"Checksum is now: {CHECKSUM_ENABLED}")

        elif choice == "4":
            break

        else:
            print("Invalid choice")

# =========================
# ENTRY
# =========================

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        manual_stdin()

    else:
        menu()
        
        
        
        ####   python -m pip install --no-index torch-2.2.2+cpu-cp310-cp310-win_amd64.whl
        ####   https://download.pytorch.org/whl/torch/
        ####   torch-2.2.2+cpu-cp310-cp310-win_amd64.whl
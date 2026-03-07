import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from fastai.tabular.all import load_learner
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


learn = load_learner("models/swing_model.pkl")
with open("models/swing_to.pkl", "rb") as f:
    to = pickle.load(f)

valid_df = to.valid.items.copy()
if valid_df.empty:
    valid_df = pd.read_csv("swing_data.csv")


dl = learn.dls.test_dl(valid_df)
learn.model.eval()
all_probs = []
with torch.no_grad():
    for b in dl:
        x_cat, x_cont = b[0], b[1]
        logits = learn.model(x_cat, x_cont)
        all_probs.append(torch.softmax(logits, dim=1).cpu())
probs = torch.cat(all_probs, dim=0).numpy()

targ_raw = valid_df["swing"]
if targ_raw.dtype == bool:
    targs = targ_raw.astype(int).to_numpy()
else:
    targs = targ_raw.astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0}).to_numpy()
    if np.isnan(targs).any():
        raise ValueError("Could not map some values in `swing` to binary labels.")
    targs = targs.astype(int)

vocab = [str(v).lower() for v in learn.dls.vocab]
if "true" in vocab:
    pos_idx = vocab.index("true")
elif "1" in vocab:
    pos_idx = vocab.index("1")
else:
    pos_idx = 1 if probs.shape[1] > 1 else 0

p_swing = probs[:, pos_idx]

brier = brier_score_loss(targs, p_swing)
print(f"Brier score: {brier:.4f}")

frac_pos, mean_pred = calibration_curve(targs, p_swing, n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot(mean_pred, frac_pos, "s-", label="Model")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction positive")
plt.title(f"Swing model calibration (Brier={brier:.4f})")
plt.legend()
plt.tight_layout()
plt.savefig("calibration_swing.png")
plt.close()
print(f"Saved reliability diagram to calibration_swing.png ({len(targs)} samples)")

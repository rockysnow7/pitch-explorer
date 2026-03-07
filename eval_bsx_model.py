import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from fastai.tabular.all import load_learner
from sklearn.calibration import calibration_curve


learn = load_learner("models/bsx_model.pkl")
with open("models/bsx_to.pkl", "rb") as f:
    to = pickle.load(f)

valid_df = to.valid.items.copy()
if valid_df.empty:
    valid_df = pd.read_csv("bsx_data.csv")

dl = learn.dls.test_dl(valid_df)
learn.model.eval()
all_probs = []
with torch.no_grad():
    for b in dl:
        x_cat, x_cont = b[0], b[1]
        logits = learn.model(x_cat, x_cont)
        all_probs.append(torch.softmax(logits, dim=1).cpu())
probs = torch.cat(all_probs, dim=0).numpy()

vocab = [str(v) for v in learn.dls.vocab]
vocab_map = {cls: i for i, cls in enumerate(vocab)}
targ_series = valid_df["type"]

if pd.api.types.is_numeric_dtype(targ_series):
    targs = targ_series.to_numpy().astype(int)
else:
    targ_raw = targ_series.astype(str)
    targs = targ_raw.map(vocab_map).to_numpy()
    if np.isnan(targs).any():
        targs_num = pd.to_numeric(targ_raw, errors="coerce").to_numpy()
        if np.isnan(targs_num).any():
            bad = sorted(set(targ_raw[pd.isna(targs)]))
            raise ValueError(f"Unknown labels in `type`: {bad}")
        targs = targs_num
    targs = targs.astype(int)

if (targs < 0).any() or (targs >= len(vocab)).any():
    raise ValueError("Found out-of-range class indices in `type`.")

one_hot = np.eye(len(vocab))[targs]
brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
print(f"Multiclass Brier score: {brier:.4f}")

fig, axes = plt.subplots(1, len(vocab), figsize=(6 * len(vocab), 5), squeeze=False)
for i, cls in enumerate(vocab):
    y_true_i = (targs == i).astype(int)
    p_i = probs[:, i]
    frac_pos, mean_pred = calibration_curve(y_true_i, p_i, n_bins=10)

    ax = axes[0, i]
    ax.plot(mean_pred, frac_pos, "s-", label=f"{cls} vs rest")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive")
    ax.set_title(f"Class {cls}")
    ax.legend()

fig.suptitle(f"BSX model calibration (Multiclass Brier={brier:.4f})")
fig.tight_layout()
plt.savefig("calibration_bsx.png")
plt.close()
print(f"Saved reliability diagram to calibration_bsx.png ({len(targs)} samples)")

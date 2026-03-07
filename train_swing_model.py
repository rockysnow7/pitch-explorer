import pickle
from pathlib import Path

import pandas as pd
from fastai.tabular.all import (
    BalancedAccuracy, Categorify, CategoryBlock, CrossEntropyLossFlat,
    EarlyStoppingCallback, Normalize, RandomSplitter,
    RocAucBinary, SaveModelCallback, default_device, set_seed,
    tabular_config, tabular_learner, TabularPandas, tensor,
)
from sklearn.metrics import classification_report


CSV_PATH = "swing_data.csv"
MODEL_DIR = Path("models")
SEED = 42
EPOCHS = 20
LR = 1e-3

set_seed(SEED)
MODEL_DIR.mkdir(exist_ok=True)


df = pd.read_csv(CSV_PATH)

CAT_COLS = ["p_throws", "stand", "inning_topbot", "prev_pitch_type", "prev_type", "pitch_type"]
NUM_COLS = [
    "inning", "bat_score", "fld_score", "outs_when_up",
    "on_3b", "on_2b", "on_1b",
    "balls", "strikes",
    "release_speed", "pfx_x", "pfx_z",
    "K%", "BB%",
    "px", "pz",
    "prev_px", "prev_pz",
    "has_prev_px", "has_prev_pz",
]


# handle NaN values for previous pitch
PREV_COLS = ["prev_px", "prev_pz"]
PREV_INDICATOR_COLS = ["has_prev_px", "has_prev_pz"]
for prev_col, indicator_col in zip(PREV_COLS, PREV_INDICATOR_COLS):
    df[indicator_col] = df[prev_col].notna().astype(float)
    df[prev_col] = df[prev_col].fillna(0.0)

for col in NUM_COLS:
    if col in PREV_COLS or col in PREV_INDICATOR_COLS:
        continue
    df[col] = df[col].fillna(df[col].median())


splits = RandomSplitter(valid_pct=0.1, seed=SEED)(df)

to = TabularPandas(
    df,
    procs = [Categorify, Normalize],
    cat_names = CAT_COLS,
    cont_names = NUM_COLS,
    y_names = "swing",
    y_block = CategoryBlock(),
    splits = splits,
)

dls = to.dataloaders(bs=8192)

alpha = 0.5
counts  = df.iloc[splits[0]]["swing"].value_counts()
weights = tensor([1 / counts[False] ** alpha, 1 / counts[True] ** alpha])
weights = (weights / weights.sum()).to(default_device())

learn = tabular_learner(
    dls,
    layers = [512, 256, 128],
    config = tabular_config(ps=[0.1, 0.1, 0.1], embed_p=0.1),
    loss_func = CrossEntropyLossFlat(weight=weights),
    metrics = [BalancedAccuracy(), RocAucBinary()],
)
learn.fit_one_cycle(
    EPOCHS, LR,
    cbs=[
        EarlyStoppingCallback("roc_auc_score", patience=5),
        SaveModelCallback("roc_auc_score"),
    ],
)


# save
learn.export(MODEL_DIR / "swing_model.pkl")

with open(MODEL_DIR / "swing_to.pkl", "wb") as f:
    pickle.dump(to, f)

print("Saved swing_model.pkl and swing_to.pkl to models/")


# eval
_, decoded, targs = learn.get_preds(with_decoded=True)
print(classification_report(targs, decoded, target_names=[str(v) for v in dls.vocab]))

import pickle
from pathlib import Path

import pandas as pd
from fastai.tabular.all import (
    BalancedAccuracy, Categorify, CategoryBlock, CrossEntropyLossFlat,
    EarlyStoppingCallback, FillMissing, Normalize, RandomSplitter,
    SaveModelCallback, default_device, set_seed, tabular_config,
    tabular_learner, TabularPandas, tensor,
)
from sklearn.metrics import classification_report


CSV_PATH = "bsx_data.csv"
MODEL_DIR = Path("models")
SEED = 42
EPOCHS = 20
LR = 1e-3

set_seed(SEED)
MODEL_DIR.mkdir(exist_ok=True)


df = pd.read_csv(CSV_PATH)

CAT_COLS = ["p_throws", "stand", "pitch_type"]
NUM_COLS = ["release_speed", "pfx_x", "pfx_z", "px", "pz", "swing"]

splits = RandomSplitter(valid_pct=0.1, seed=SEED)(df)

to = TabularPandas(
    df,
    procs = [Categorify, FillMissing, Normalize],
    cat_names = CAT_COLS,
    cont_names = NUM_COLS,
    y_names = "type",
    y_block = CategoryBlock(),
    splits = splits,
)

dls = to.dataloaders(bs=8192)

counts  = df.iloc[splits[0]]["type"].value_counts()
weights = tensor([1 / counts["B"], 1 / counts["S"], 1 / counts["X"]])
weights = (weights / weights.sum()).to(default_device())

learn = tabular_learner(
    dls,
    layers = [512, 256, 128],
    config = tabular_config(ps=[0.3, 0.3, 0.3], embed_p=0.1),
    loss_func = CrossEntropyLossFlat(weight=weights),
    metrics = [BalancedAccuracy()],
)
learn.fit_one_cycle(
    EPOCHS, LR,
    cbs=[
        EarlyStoppingCallback("balanced_accuracy_score", patience=5),
        SaveModelCallback("balanced_accuracy_score"),
    ],
)


# save
learn.export(MODEL_DIR / "bsx_model.pkl")

with open(MODEL_DIR / "bsx_to.pkl", "wb") as f:
    pickle.dump(to, f)

print("Saved bsx_model.pkl and bsx_to.pkl to models/")

# eval
_, decoded, targs = learn.get_preds(with_decoded=True)
print(classification_report(targs, decoded, target_names=dls.vocab))

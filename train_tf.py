from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
from tensorflow import keras
from tensorflow.keras import layers

CSV_PATH = Path("data/diabetic_data.csv")
MODEL_PATH = Path("tf_model.h5")
PREPROC_PATH = Path("tf_preprocessor.joblib")

CAT_COLS = ["race", "gender", "age", "A1Cresult", "insulin", "change", "diabetesMed"]
NUM_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient"
]

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["?", "NA", "NaN"])
    df = df[df["readmitted"].isin(["<30", ">30", "NO"])].copy()
    df["target"] = (df["readmitted"] != "NO").astype(int)
    return df[CAT_COLS + NUM_COLS + ["target"]]

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", Pipeline([("scale", StandardScaler())]), NUM_COLS),
        ]
    )

def build_model(input_dim: int):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model

def main():
    df = load_data(CSV_PATH)
    X, y = df.drop(columns=["target"]), df["target"]
    pre = build_preprocessor()
    Xp = pre.fit_transform(X)
    dump(pre, PREPROC_PATH)
    Xtr, Xte, ytr, yte = train_test_split(Xp, y, test_size=0.2, random_state=42, stratify=y)
    model = build_model(Xtr.shape[1])
    model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=10, batch_size=32, verbose=1)
    model.save(MODEL_PATH)
    print(f"✅ Saved model: {MODEL_PATH.resolve()}")
    print(f"✅ Saved preprocessor: {PREPROC_PATH.resolve()}")

if __name__ == "__main__":
    main()

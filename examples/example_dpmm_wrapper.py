import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from spaceai.models.anomaly.dpmm_detector import DPMMWrapperDetector
from spaceai.external.dpmm_wrapper.benchmark_utils import evaluate_metrics, set_seed_numpy

<<<<<<< HEAD

# Carico il dataset
=======
# -----------------------
# Caricamento del dataset
# -----------------------
>>>>>>> 57fe8dc (versione con dpmm)
df = pd.read_csv("dataset.csv", index_col="segment")

features = [
    "mean", "var", "std", "len", "duration", "len_weighted", "gaps_squared", "n_peaks",
    "smooth10_n_peaks", "smooth20_n_peaks", "var_div_duration", "var_div_len",
    "diff_peaks", "diff2_peaks", "diff_var", "diff2_var", "kurtosis", "skew"
]

X_train_nominal = df.loc[(df.anomaly == 0) & (df.train == 1), features]
X_test = df.loc[df.train == 0, features]
y_test = df.loc[df.train == 0, "anomaly"]

# -----------------------
# Preprocessing
# -----------------------
scaler = StandardScaler()
X_train_nominal_scaled = scaler.fit_transform(X_train_nominal)
X_test_scaled = scaler.transform(X_test)

set_seed_numpy(2137)

# -----------------------
<<<<<<< HEAD
# Inizializzo e uso il detector
=======
# Inizializzo il detector
>>>>>>> 57fe8dc (versione con dpmm)
# -----------------------
detector = DPMMWrapperDetector(
    mode="likelihood",      # oppure "new_cluster"
    model_type="Full",
    K=100,
    num_iterations=50,
    lr=0.8,
<<<<<<< HEAD
    python_executable="C:/Users/manet/anaconda3/envs/dpmm_env/python.exe"  # Inserisci il percorso corretto del tuo ambiente Python
)

y_pred = detector(X_test_scaled, y_test.values, X_train_nominal=X_train_nominal_scaled)

# -----------------------
# Valuto le performance
# -----------------------
metrics = evaluate_metrics(y_test.values, y_pred)

print("\nRisultati del DPMM Wrapper:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
=======
    python_executable="C:/Users/manet/anaconda3/envs/dpmm_env/python.exe"  # <- cambia se necessario
)

# -----------------------
# Metodo separato -> fit + predict
# -----------------------
detector.fit(X_train_nominal_scaled)
y_pred_sep = detector.predict(X_test_scaled)
metrics_sep = evaluate_metrics(y_test.values, y_pred_sep)

for k, v in metrics_sep.items():
    print(f"{k} : {v:.4f}")



>>>>>>> 57fe8dc (versione con dpmm)

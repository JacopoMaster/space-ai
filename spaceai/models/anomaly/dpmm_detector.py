import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
import shutil
import sys
from .detector import AnomalyDetector

class DPMMWrapperDetector(AnomalyDetector):
    def __init__(self, mode="likelihood", model_type="Full", K=100, num_iterations=100, lr=0.8, python_executable=None):
        super().__init__()
        self.mode = mode
        self.model_type = model_type
        self.K = K
        self.num_iterations = num_iterations
        self.lr = lr
        self.python_executable = python_executable or shutil.which("python")

<<<<<<< HEAD
    def __call__(self, input: np.ndarray, y_true: np.ndarray, **kwargs) -> np.ndarray:
        return self.detect_anomalies(input, y_true, **kwargs)

    def detect_anomalies(self, y_pred, y_true, **kwargs):
        X_train_nominal = kwargs.get("X_train_nominal")

        # Verifica ambiente Python compatibile con DPMM
        active_env = os.environ.get("CONDA_DEFAULT_ENV", "(non rilevato)")
        print(f"Ambiente attivo: {active_env}")
        print(f"Python interpreter in uso: {self.python_executable}\n")

        if "dpmm" not in self.python_executable.lower():
            raise RuntimeError(
                f"Python interpreter non compatibile: {self.python_executable}\n"
                "Devi usare l'interprete dell'ambiente `dpmm_env` per eseguire correttamente il wrapper.\n"
                "Passalo nel costruttore con: DPMMWrapperDetector(..., python_executable='path/dpmm_env/python')\n"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_test = os.path.join(tmpdir, "test.csv")
            input_train = os.path.join(tmpdir, "train.csv")
            output_pred = os.path.join(tmpdir, "output.csv")

            pd.DataFrame(y_pred).to_csv(input_test, index=False)
            pd.DataFrame(X_train_nominal).to_csv(input_train, index=False)

            this_dir = os.path.dirname(__file__)
            run_dpmm_path = os.path.abspath(
                os.path.join(this_dir, "../../../spaceai/external/dpmm_wrapper/run_dpmm.py")
            )

            try:
                result = subprocess.run([
                    self.python_executable,
                    run_dpmm_path,
                    input_test,
                    input_train,
                    output_pred,
                    self.mode,
                    self.model_type,
                    str(self.K),
                    str(self.num_iterations),
                    str(self.lr)
                ], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print("\n🚨 ERRORE NEL SUBPROCESS DPMM:")
                print("🔹 STDOUT:")
                print(e.stdout)
                print("🔹 STDERR:")
                print(e.stderr)
                raise

            pred_df = pd.read_csv(output_pred)
            return pred_df["prediction"].values
=======
        # Crea una cartella temporanea per train/test/model
        self._tempdir = tempfile.mkdtemp()
        self._train_path = os.path.join(self._tempdir, "train.csv")
        self._test_path = os.path.join(self._tempdir, "test.csv")
        self._output_path = os.path.join(self._tempdir, "output.csv")
        self._model_path = os.path.join(self._tempdir, "model.pkl")
        self._clusters_path = os.path.join(self._tempdir, "clusters.json")

    def __call__(self, input: np.ndarray, y_true: np.ndarray, **kwargs) -> np.ndarray:
        return self.predict(input)

    def fit(self, X):
        """Esegue il fit separato tramite subprocess e salva il modello e cluster."""
        pd.DataFrame(X).to_csv(self._train_path, index=False)

        this_dir = os.path.dirname(__file__)
        run_dpmm_path = os.path.abspath(
            os.path.join(this_dir, "../../../spaceai/external/dpmm_wrapper/run_dpmm.py")
        )

        print(f"\nEseguo FIT DPMM in ambiente: {self.python_executable}")

        try:
            subprocess.run([
                self.python_executable,
                run_dpmm_path,
                "--mode", "fit",
                "--train", self._train_path,
                "--model", self._model_path,
                "--clusters", self._clusters_path,
                "--model_type", self.model_type,
                "--K", str(self.K),
                "--iterations", str(self.num_iterations),
                "--lr", str(self.lr)
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("\n🚨 ERRORE DURANTE FIT:")
            print("🔹 STDOUT:\n", e.stdout)
            print("🔹 STDERR:\n", e.stderr)
            raise

    def predict(self, X):
        """Esegue il predict separato tramite subprocess usando il modello salvato."""
        pd.DataFrame(X).to_csv(self._test_path, index=False)

        this_dir = os.path.dirname(__file__)
        run_dpmm_path = os.path.abspath(
            os.path.join(this_dir, "../../../spaceai/external/dpmm_wrapper/run_dpmm.py")
        )

        print(f"\nEseguo PREDICT DPMM in ambiente: {self.python_executable}")

        try:
            subprocess.run([
                self.python_executable,
                run_dpmm_path,
                "--mode", "predict",
                "--test", self._test_path,
                "--model", self._model_path,
                "--clusters", self._clusters_path,
                "--output", self._output_path
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("\n🚨 ERRORE DURANTE PREDICT:")
            print("🔹 STDOUT:\n", e.stdout)
            print("🔹 STDERR:\n", e.stderr)
            raise

        pred_df = pd.read_csv(self._output_path)
        return pred_df["prediction"].values

    def detect_anomalies(self, X, y_true=None, **kwargs):
        """Compatibilità: detect_anomalies richiama semplicemente predict."""
        return self.predict(X)
>>>>>>> 57fe8dc (versione con dpmm)

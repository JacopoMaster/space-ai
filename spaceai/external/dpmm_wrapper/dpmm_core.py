import numpy as np
import torch as th
import torch.optim as optim
from torch_dpmm.models import FullGaussianDPMM, DiagonalGaussianDPMM, SingleGaussianDPMM, UnitGaussianDPMM
from tqdm import tqdm

<<<<<<< HEAD
=======
# Inizializza il modello DPMM a seconda del tipo richiesto

>>>>>>> 57fe8dc (versione con dpmm)
def _init_model(model_type, K, D, alphaDP=10):
    if model_type == "Full":
        return FullGaussianDPMM(K, D, alphaDP, mu_prior=0, mu_prior_strength=0.01, var_prior=0.1, var_prior_strength=100)
    elif model_type == "Diagonal":
        return DiagonalGaussianDPMM(K, D, alphaDP, mu_prior=0, mu_prior_strength=0.01, var_prior=0.1, var_prior_strength=100)
    elif model_type == "Single":
        return SingleGaussianDPMM(K, D, alphaDP, mu_prior=0, mu_prior_strength=0.01, var_prior=0.1, var_prior_strength=100)
    elif model_type == "Unit":
        return UnitGaussianDPMM(K, D, alphaDP, mu_prior=0, mu_prior_strength=0.01)
    else:
        raise ValueError("Invalid model_type")

<<<<<<< HEAD
=======
# Metodo "vecchio stile" che esegue fit+predict con soglia su log-likelihood

>>>>>>> 57fe8dc (versione con dpmm)
def run_dpmm_likelihood(X_train_nominal, X_test, model_type="Full", K=100, num_iterations=100, lr=0.8):
    D = X_train_nominal.shape[1]
    dpmm_model = _init_model(model_type, K, D)

    X_train_tensor = th.tensor(X_train_nominal, dtype=th.float32)
    X_test_tensor = th.tensor(X_test, dtype=th.float32)

    dpmm_model.init_var_params(X_train_tensor)
    optimizer = optim.SGD(dpmm_model.parameters(), lr=lr)

    for epoch in tqdm(range(num_iterations), desc=f"Training {model_type} DPMM", unit="epoch"):
        optimizer.zero_grad()
        _, elbo_loss, _ = dpmm_model(X_train_tensor)
        elbo_loss.backward()
        optimizer.step()

    with th.no_grad():
        _, _, train_log_likelihood = dpmm_model(X_train_tensor)
        anomaly_threshold = np.percentile(train_log_likelihood.numpy(), 5)

        _, _, test_log_likelihood = dpmm_model(X_test_tensor)
        y_pred = (test_log_likelihood.numpy() < anomaly_threshold).astype(int)

    return y_pred

<<<<<<< HEAD
=======
# Metodo "vecchio stile" che esegue fit+predict con logica dei cluster assegnati

>>>>>>> 57fe8dc (versione con dpmm)
def run_dpmm_new_cluster(X_train_nominal, X_test, model_type="Full", K=100, num_iterations=100, lr=0.8, resp_threshold=0.1, min_cluster_weight=5.0):
    D = X_train_nominal.shape[1]
    dpmm_model = _init_model(model_type, K, D)

    X_train_tensor = th.tensor(X_train_nominal, dtype=th.float32)
    X_test_tensor = th.tensor(X_test, dtype=th.float32)

    dpmm_model.init_var_params(X_train_tensor)
    optimizer = optim.SGD(dpmm_model.parameters(), lr=lr)

    for epoch in tqdm(range(num_iterations), desc=f"Training {model_type} DPMM", unit="epoch"):
        optimizer.zero_grad()
        _, elbo_loss, _ = dpmm_model(X_train_tensor)
        elbo_loss.backward()
        optimizer.step()

    with th.no_grad():
        r_train, _, _ = dpmm_model(X_train_tensor)
        cluster_weight = th.sum(r_train, dim=0).numpy()
        active_clusters = np.where(cluster_weight > min_cluster_weight)[0].tolist()

        r_test, _, _ = dpmm_model(X_test_tensor)
        assigned_clusters = th.argmax(r_test, dim=1).numpy()
        max_resp = th.max(r_test, dim=1).values.numpy()

        y_pred = np.array([
            1 if (cluster not in active_clusters or max_resp[i] < resp_threshold) else 0
            for i, cluster in enumerate(assigned_clusters)
        ])

<<<<<<< HEAD
    return y_pred
=======
    return y_pred

# Nuova versione: restituisce solo il modello allenato (usato in modalità fit)

def get_trained_dpmm_model(X_train, model_type="Full", K=100, num_iterations=100, lr=0.8):
    D = X_train.shape[1]
    dpmm_model = _init_model(model_type, K, D)

    X_train_tensor = th.tensor(X_train, dtype=th.float32)
    dpmm_model.init_var_params(X_train_tensor)
    optimizer = optim.SGD(dpmm_model.parameters(), lr=lr)

    for epoch in tqdm(range(num_iterations), desc=f"Fitting {model_type} DPMM", unit="epoch"):
        optimizer.zero_grad()
        _, elbo_loss, _ = dpmm_model(X_train_tensor)
        elbo_loss.backward()
        optimizer.step()

    return dpmm_model

# Nuova versione: predizione da modello salvato, con soglia log-likelihood

def run_dpmm_likelihood(model, X_test):
    X_test_tensor = th.tensor(X_test, dtype=th.float32)
    with th.no_grad():
        _, _, test_log_likelihood = model(X_test_tensor)
        threshold = np.percentile(test_log_likelihood.numpy(), 5)
        return (test_log_likelihood.numpy() < threshold).astype(int)

# Nuova versione: predizione da modello salvato, con logica cluster assegnati

def run_dpmm_new_cluster(model, X_test, active_clusters, resp_threshold=0.1):
    X_test_tensor = th.tensor(X_test, dtype=th.float32)
    with th.no_grad():
        r_test, _, _ = model(X_test_tensor)
        assigned_clusters = th.argmax(r_test, dim=1).numpy()
        max_resp = th.max(r_test, dim=1).values.numpy()

        return np.array([
            1 if (cluster not in active_clusters or max_resp[i] < resp_threshold) else 0
            for i, cluster in enumerate(assigned_clusters)
        ])
>>>>>>> 57fe8dc (versione con dpmm)

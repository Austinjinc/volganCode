import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.dates as mdates
import VolGAN
import pandas as pd
import os
from scipy.stats import norm

job_id = os.environ.get('SLURM_JOB_ID', 'default')
task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
job_dir = f"output_{job_id}"
task_dir = os.path.join(job_dir, f"task_{task_id}")
os.makedirs(task_dir, exist_ok=True)
print(f"Output directory created: {task_dir}")

datapath = "heston_market_data.csv"
surfacepath = "heston_iv_surface.csv"

# Use the helper function `SPXData` to load raw data and implied vol surfaces
# SPXData function is MODIFIED
(
    surfaces_transform,
    prices,
    prices_prev,
    log_rtn,
    m,
    tau,
    ms,
    taus,
    dates_dt,
) = VolGAN.SPXData(datapath, surfacepath)

# -----------------------------------------------------------------
# Prepare input features for the GAN

dates_t = dates_dt[22:]
log_rtn_t = log_rtn[22:]
log_rtn_tm1 = log_rtn[21:-1]
log_rtn_tm2 = log_rtn[20:-2]
log_iv_t = np.log(surfaces_transform[22:])
log_iv_tm1 = np.log(surfaces_transform[21:-1])
log_iv_inc_t = log_iv_t - log_iv_tm1

# ---------------------------------------------------------------------------
# Training hyperparameters

tr = 0.95         # proportion of data used for training
noise_dim = 32    # dimension of latent noise vector
hidden_dim = 16   # width of generator/discriminator hidden layers
n_epochs = 10000  # number of epochs for the main training loop
n_grad = 25       # number of gradient-matching iterations

device = "cpu"

# -------------------------------------------------------------------------
# Train the model using the high level interface in VolGAN.py
# VolGAN returns trained networks and the split datasets.

(
    gen,
    gen_opt,
    disc,
    disc_opt,
    true_train,
    true_val,
    true_test,
    condition_train,
    condition_val,
    condition_test,
    dates_t,
    m,
    tau,
    ms,
    taus
) = VolGAN.VolGAN(
    datapath=datapath,
    surfacepath=surfacepath,
    tr=tr,
    noise_dim=noise_dim,
    hidden_dim=hidden_dim,
    n_epochs=n_epochs,
    n_grad=n_grad,
    lrg=0.0001,
    lrd=0.0001,
    batch_size=100,
    device=device,
)

# Save the trained model
job_id = os.environ.get('SLURM_JOB_ID', 'default')
task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
job_dir = f"output_{job_id}"
task_dir = os.path.join(job_dir, f"task_{task_id}")
os.makedirs(task_dir, exist_ok=True)
torch.save(
    {
        "gen_state": gen.state_dict(),
        "disc_state": disc.state_dict(),
    },
    os.path.join(task_dir, "volgan_model.pth")
)


n_test = true_test.shape[0]  # number of test days
B = 100                      # number of samples per day to generate

# Pre-compute penalty matrices used to evaluate arbitrage violations
mP_t, mP_k, mPb_K = VolGAN.penalty_mutau_tensor(m, tau, device)

# Convert moneyness and tau grids to tensors
m_t = torch.tensor(ms, dtype=torch.float, device=device)
t_t = torch.tensor(taus, dtype=torch.float, device=device)

# Storage for generated surfaces and penalties
gen_surfs = np.zeros((B, n_test, 81))
gen_returns = np.zeros((B, n_test))


# Disable gradient tracking during evaluation
with torch.no_grad():
    for l in tqdm(range(B)):
        # Sample latent noise
        noise = torch.randn((n_test, noise_dim), device=device, dtype=torch.float)
        # Generate log-return and surface increment conditioned on past data
        fake = gen(noise, condition_test)
        # Previous day's surface
        surface_past_test = condition_test[:, 3:]
        # Generated surface in original scale
        fake_surface = torch.exp(fake[:, 1:] + surface_past_test)
        # Store generated surfaces and returns
        gen_surfs[l] = fake_surface.cpu().numpy()
        gen_returns[l] = fake[:, 0].cpu().numpy()


# np.save(f"generated_surfaces_{job_id}.npy", gen_surfs)
# np.save(f"generated_returns_{job_id}.npy", gen_returns)



def penalty_mutau(mu, T):
    P_T = np.zeros((len(T), len(T)))
    P_K = np.zeros((len(mu), len(mu)))
    PB_K = np.zeros((len(mu), len(mu)))
    for j in np.arange(0, len(T)-1, 1):
        P_T[j, j] = T[j]/(T[j+1]-T[j])
        P_T[j+1, j] = -T[j]/(T[j+1]-T[j])
    for i in np.arange(0, len(mu)-1, 1):
        P_K[i, i] = -1/(mu[i+1]-mu[i])
        P_K[i, i+1] = 1/(mu[i+1]-mu[i])
    for i in np.arange(1, len(mu)-1, 1):
        PB_K[i, i-1] = -(mu[i+1]-mu[i]) / ((mu[i]-mu[i-1]) * (mu[i+1]-mu[i]))
        PB_K[i, i] = (mu[i+1] - mu[i-1]) / ((mu[i]-mu[i-1]) * (mu[i+1]-mu[i]))
        PB_K[i, i+1] = -(mu[i]-mu[i-1]) / ((mu[i]-mu[i-1]) * (mu[i+1]-mu[i]))
    return P_T, P_K, PB_K

def arbitrage_penalty(C, P_T, P_K, PB_K):
    P1 = np.maximum(0, np.matmul(C, P_T))
    P2 = np.maximum(0, np.matmul(P_K, C))
    P3 = np.maximum(0, np.matmul(PB_K, C))
    return P1, P2, P3, np.sum(P1+P2+P3)

def smallBS(m, tau, sigma, r):
    d1 = (-np.log(m) + tau*(r+0.5*sigma*sigma)) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    price = norm.cdf(d1) - m*norm.cdf(d2)*np.exp(-r*tau)
    return price

def compute_total_ap(sample_vec, m_grid, ttm_grid, r, P_T, P_K, PB_K):
    sample_data = np.array(sample_vec).reshape(9, 9)
    M, T = np.meshgrid(m_grid, ttm_grid)
    price_matrix = smallBS(M, T, sample_data, r)
    _, _, _, total_ap = arbitrage_penalty(price_matrix.T, P_T, P_K, PB_K)
    return total_ap

def smallBS_batch(m_grid, ttm_grid, sigma_batch, r):
    """
    Vectorized Black-Scholes price calculation over:
    - Multiple samples (S)
    - Multiple days (B)
    - Flattened implied vol surfaces (N = lt * lk)

    Args:
        m_grid: (lk,) array of moneyness values
        ttm_grid: (lt,) array of time-to-maturity values
        sigma_batch: (S, B, N) flattened IV surfaces
        r: risk-free rate

    Returns:
        prices: (S, B, lt, lk) array of option prices
    """
    S, B, N = sigma_batch.shape
    lt, lk = len(ttm_grid), len(m_grid)
    assert N == lt * lk, f"Expected flattened surface of length {lt * lk}, got {N}"

    # Reshape IV surfaces
    sigma = sigma_batch.reshape(S, B, lt, lk)  # shape (S, B, lt, lk)

    # Broadcast moneyness and maturity
    m = m_grid[None, None, None, :]           # (1, 1, 1, lk)
    tau = ttm_grid[None, None, :, None]       # (1, 1, lt, 1)

    # Avoid divide-by-zero issues
    sigma = np.clip(sigma, 1e-6, None)
    sqrt_tau = np.sqrt(tau)

    # Black-Scholes d1, d2
    d1 = (-np.log(m) + tau * (r + 0.5 * sigma**2)) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    # Call price (forward moneyness form)
    prices = norm.cdf(d1) - m * norm.cdf(d2) * np.exp(-r * tau)

    return prices  # shape: (S, B, lt, lk)

def arbitrage_penalty_batch(prices_batch, P_T, P_K, PB_K):
    P1 = np.maximum(0, np.matmul(prices_batch.transpose(0, 1, 3, 2), P_T))

    P2 = np.maximum(0, np.matmul(P_K, prices_batch.transpose(0, 1, 3, 2)))

    P3 = np.maximum(0, np.matmul(PB_K, prices_batch.transpose(0, 1, 3, 2)))

    # Sum over lt and lk to get scalar penalty per surface
    total_penalty = np.sum(P1 + P2 + P3, axis=(2, 3))  # (S, B)

    return total_penalty
# -------- SETUP AND FILE LOAD ----------
surf_data = pd.read_csv('heston_iv_surface.csv', header=None)
surf_data = surf_data.tail(n_test)  # keep last 890 rows for alignment

m_grid = np.array([0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3])
ttm_grid = np.array([1/12, 1/6, 1/4, 1/3, 1/2, 3/4, 1.0, 1.5, 2])
r = 0
P_T, P_K, PB_K = penalty_mutau(m_grid, ttm_grid)

# --- GET JOB ID, CREATE DIR ---

# --------- PENALTIES: GENERATED DATA -----------
prices = smallBS_batch(m_grid, ttm_grid, gen_surfs, 0)
penalties = arbitrage_penalty_batch(prices, P_T, P_K, PB_K)

percentiles = [5, 95]
rows_percentiles = np.percentile(penalties, percentiles, axis=0)  # (4, 890)
rows_mean = np.mean(penalties, axis=0)  # (890,)

summary = np.hstack([rows_percentiles.T, rows_mean[:, None]])  # (890, 5)
summary_df = pd.DataFrame(summary, columns=['p5', 'p95', 'mean'])

# --------- PENALTIES: REAL DATA -----------
real_penalties = []
for i in range(surf_data.shape[0]):
    sample_vec = surf_data.iloc[i, 1:].values  # skip date column
    sample_data = np.array(sample_vec, dtype=float).reshape(9, 9)
    M, T = np.meshgrid(m_grid, ttm_grid)
    price_matrix = smallBS(M, T, sample_data, r)
    _, _, _, total_ap = arbitrage_penalty(price_matrix.T, P_T, P_K, PB_K)
    real_penalties.append(total_ap)
real_penalties = np.array(real_penalties)  # (890,)

# --------- DATES FOR X-AXIS -----------
dates = pd.to_datetime(surf_data.iloc[:, 0].values)

# --------- PENALTY PLOT -----------
columns_to_plot = ['p5', 'p95']
plt.figure(figsize=(20, 10))
for col in columns_to_plot:
    plt.plot(dates, summary_df[col], label=col)
plt.scatter(dates, real_penalties, label='Real', color='red', marker='.', s=20)
plt.xlabel('Date')
plt.ylabel('Arbitrage Penalty')
plt.title('Arbitrage Penalty: Generated Percentiles vs Real Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{task_dir}/penalty_percentiles_{job_id}.png")
plt.close()

#--------- ATM IV plots --------------
atm_indices = {
    '1m': 5,
    '2m': 15,
    '3m': 35,
    '1y': 75
}
percentiles = [2.5, 5, 95, 97.5]
maturity_labels = {'1m': '1m ATM', '2m': '2m ATM', '3m': '3m ATM', '1y': '1y ATM'}
for key, idx in atm_indices.items():
    atm_data = gen_surfs[:, :, idx-1]  # shape: (10000, 890)
    atm_percentiles = np.percentile(atm_data, percentiles, axis=0)  # (4, 890)
    real_iv = surf_data.iloc[:, idx].values  # col 0 = date, col idx = correct IV
    plt.figure(figsize=(20, 10))
    plt.plot(dates, atm_percentiles[0], label='2.5th Percentile', linestyle=':')
    plt.plot(dates, atm_percentiles[1], label='5th Percentile', linestyle='--')
    plt.plot(dates, atm_percentiles[2], label='95th Percentile', linestyle='--')
    plt.plot(dates, atm_percentiles[3], label='97.5th Percentile', linestyle=':')
    plt.scatter(dates, real_iv, label='Real', marker='.', color='red')
    plt.xlabel('Date')
    plt.ylabel(f'{maturity_labels[key]} IV')
    plt.title(f'{maturity_labels[key]}: 2.5th, 5th, 95th, 97.5th Percentiles and Real Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{task_dir}/atm_{key}_{job_id}.png")
    plt.close()

# -------------- OTM plots ------------------
otm_indices = {
    '1m OTM': 8,
    '2m OTM': 18,
    '3m OTM': 38,
    '1y OTM': 78
}
percentiles = [5, 95]
for label, i in otm_indices.items():
    otm_data = gen_surfs[:, :, i-1]  # (10000, 890)
    otm_percentiles = np.percentile(otm_data, percentiles, axis=0)  # (2, 890)
    real_otm = surf_data.iloc[:, i].values  # i-th column is correct IV
    plt.figure(figsize=(20, 10))
    plt.plot(dates, otm_percentiles[0], label='5th Percentile')
    plt.plot(dates, otm_percentiles[1], label='95th Percentile')
    plt.scatter(dates, real_otm, label='Real', marker='.', color='red')
    plt.xlabel('Date')
    plt.ylabel(f'{label} IV')
    plt.title(f'{label}: 5th, 95th Percentiles and Real Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{task_dir}/otm_{label.replace(' ', '_')}_{job_id}.png")
    plt.close()

# -------------- ITM plots ----------------
itm_indices = {
    '1m ITM': 2,
    '2m ITM': 12,
    '3m ITM': 32,
    '1y ITM': 72
}
percentiles = [5, 95]
for label, i in itm_indices.items():
    itm_data = gen_surfs[:, :, i-1]  # (10000, 890)
    itm_percentiles = np.percentile(itm_data, percentiles, axis=0)  # (2, 890)
    real_itm = surf_data.iloc[:, i].values  # i-th column is correct IV
    plt.figure(figsize=(20, 10))
    plt.plot(dates, itm_percentiles[0], label='5th Percentile')
    plt.plot(dates, itm_percentiles[1], label='95th Percentile')
    plt.scatter(dates, real_itm, label='Real', marker='.', color='red')
    plt.xlabel('Date')
    plt.ylabel(f'{label} IV')
    plt.title(f'{label}: 5th, 95th Percentiles and Real Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{task_dir}/itm_{label.replace(' ', '_')}_{job_id}.png")
    plt.close()

selected_days = np.linspace(0, gen_surfs.shape[1] - 1, 5, dtype=int)
selected_samples = np.linspace(0, gen_surfs.shape[0] - 1, 5, dtype=int)

M, T = np.meshgrid(m_grid, ttm_grid)

for day_idx in selected_days:
    for sample_idx in selected_samples:
        surface = gen_surfs[sample_idx, day_idx].reshape(9, 9)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(M, T, surface, cmap='viridis')
        ax.set_title(f"Surface | Day {day_idx} | Sample {sample_idx}")
        ax.set_xlabel('Moneyness')
        ax.set_ylabel('TTM (years)')
        ax.set_zlabel('IV')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()

        filename = f"surface_day{day_idx}_sample{sample_idx}_{job_id}.png"
        plt.savefig(os.path.join(task_dir, filename))
        plt.close()
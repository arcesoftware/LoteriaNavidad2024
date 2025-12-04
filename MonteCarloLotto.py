import numpy as np
import time
from collections import Counter

# ------------------------
# Simple generator assumptions (adjust if you want)
# ------------------------
# We'll use a Gaussian perturbation around a mean-reversion to mu
np.random.seed(0)

N = 1_000_000  # number of MonteCarlo trials
last_value = 43         # 2024 value (known)
mu = 47.9               # long-run mean from our calibration
sigma = 23.0            # per-year volatility (calibrated)

# simple mean-reverting one-step model:
# next = last_value + alpha*(mu - last_value) + eps, eps~N(0, sigma_eps^2)
# choose alpha to reflect some reversion to mean; alpha in [0..1]
alpha = 0.25
sigma_eps = sigma * np.sqrt(1 - (1 - alpha)**2)  # keep marginal variance ~sigma
# But simplest: next = (1-alpha)*last + alpha*mu + Normal(0, sigma_eps)
mean_shift = (1 - alpha) * last_value + alpha * mu

t0 = time.time()
# vectorized sampling
samples = np.random.normal(loc=mean_shift, scale=sigma_eps, size=N)
# round to integers and map to 0..99
samples_int = np.mod(np.round(samples).astype(int), 100)

# compute stats
mean_sim = samples_int.mean()
median_sim = np.median(samples_int)
std_sim = samples_int.std(ddof=0)
p05 = np.percentile(samples_int, 5)
p95 = np.percentile(samples_int, 95)
counts = Counter(samples_int)
top_modes = counts.most_common(10)

t1 = time.time()

print(f"Simulations: {N:,}, time {t1-t0:.2f}s")
print("Mean:", mean_sim)
print("Median:", median_sim)
print("Std :", std_sim)
print("5%  :", p05, " 95%:", p95)
print("Top 10 most frequent integers:", top_modes[:10])

# bucket histogram (0-9, 10-19, ..., 90-99)
bins = [f"{i:02d}-{i+9:02d}" for i in range(0,100,10)]
hist, _ = np.histogram(samples_int, bins=np.arange

import matplotlib.pyplot as plt
from datetime import datetime
from torch import tensor
import numpy as np
import torch
from itertools import product


def kl_2d(target_dist, empirical_dist):
    target_dist = np.clip(target_dist, 1e-12, None)
    empirical_dist = np.clip(empirical_dist, 1e-12, None)
    kl_div = np.sum(target_dist * np.log(target_dist / empirical_dist))
    return kl_div


def rbf_kernel(X, Y, sigma=1.0):
    gamma = 1 / (2 * sigma ** 2)
    XX = np.sum(X ** 2, axis=1)[:, np.newaxis]
    YY = np.sum(Y ** 2, axis=1)[np.newaxis, :]
    distances = XX + YY - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * distances)


def random_fourier_features(X, D, sigma=1.0):
    d = X.shape[1]
    W = np.random.normal(0, 1 / sigma, size=(D, d))
    b = np.random.uniform(0, 2 * np.pi, size=D)
    return np.sqrt(2 / D) * np.cos(np.dot(X, W.T) + b)


def mmd_2d(X, Y, D=1000, sigma=1.0):
    X_features = random_fourier_features(X, D, sigma)
    Y_features = random_fourier_features(Y, D, sigma)

    mu_X = np.mean(X_features, axis=0)
    mu_Y = np.mean(Y_features, axis=0)

    return np.linalg.norm(mu_X - mu_Y) ** 2

def mmd_dim(X, Y, D=1000, sigma=1.0, random_state=None):
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError(f"X and Y must be 2D tensors, but got X.dim()={X.dim()}, Y.dim()={Y.dim()}")

    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Dimension mismatch: X has dim={X.shape[1]}, Y has dim={Y.shape[1]}")

    dim = X.shape[1]

    if random_state is not None:
        torch.manual_seed(random_state)

    W = torch.normal(mean=0.0, std=1.0/sigma, size=(dim, D), device=X.device)
    b = torch.rand(D, device=X.device) * 2 * torch.pi

    projection_X = torch.matmul(X, W) + b  # (n_X, D)
    projection_Y = torch.matmul(Y, W) + b  # (n_Y, D)

    Z_X = torch.sqrt(2.0 / D) * torch.cos(projection_X)  # (n_X, D)
    Z_Y = torch.sqrt(2.0 / D) * torch.cos(projection_Y)  # (n_Y, D)

    mu_X = torch.mean(Z_X, dim=0)  # (D,)
    mu_Y = torch.mean(Z_Y, dim=0)  # (D,)

    mmd_val = torch.norm(mu_X - mu_Y).item() ** 2

    return mmd_val

def difference_function(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        x_pert = x.clone()
        x_pert[:, i] = 1.0 - x[:, i]
        delta = model(x_pert).squeeze() - orig_out
        d[:, i] = delta
    return d


def set_rand_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calc_probs1(ss_dim, energy_function, device):
    energy_function.one_hot = False
    samples = []
    for i in range(ss_dim):
        for j in range(ss_dim):
            samples.append((i, j))
    samples = tensor(samples).to(device)
    energies = energy_function(samples)
    #z = np.exp(energies).sum()
    z = np.exp(energies.cpu().numpy()).sum()
    #z = energies.exp().sum()

    probs = energies.exp() / z
    return samples, probs
    
def calc_probs(dim, ss_dim, energy_function, device):
    energy_function.one_hot = False
    ranges = [range(ss_dim) for _ in range(dim)]
    samples = list(product(*ranges))
    samples_tensor = torch.tensor(samples, dtype=torch.float32).to(device)
    energies = energy_function(samples_tensor)
    max_energy = energies.max()
    stable_energies = energies - max_energy
    z = torch.exp(stable_energies).sum().item()

    probs = torch.exp(stable_energies) / z

    return samples_tensor, probs

def get_time():
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")


def plot_target(dist_img, cur_dir, task):
    plt.figure(figsize=(8, 8), dpi=200)
    plt.contourf(dist_img, levels=100, cmap=plt.cm.coolwarm)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{cur_dir}/target_" + task + ".png", bbox_inches="tight", transparent=True, pad_inches=0)
    plt.close()
    print(f"gt: {cur_dir}/init_dist.png")


def plot_empirical(est_img, cur_dir, args, sampler_name, task, mmd, kl):
    plt.figure(figsize=(8, 8), dpi=200)
    plt.contourf(est_img, levels=100, cmap=plt.cm.coolwarm)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        f"{cur_dir}/empirical" + task + "_" + sampler_name + "_step_" + str(args.step1) + "_"
        + str(args.sampling_steps) + "_mmd_" + str(mmd) + "_kl_" + str(kl) + ".png", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Save empirical dist: {cur_dir}/est_dist" + task + "_" + sampler_name
          + "_step_" + str(args.step1) + "_" + str(args.sampling_steps)  + ".png")

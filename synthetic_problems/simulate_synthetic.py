from utils import kl_2d, mmd_2d, calc_probs1, set_rand_seeds, plot_target, plot_empirical
from samplers import run_sampler, get_sampler
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import os
import argparse
from scipy.interpolate import PchipInterpolator
from scipy.optimize import bisect
import math
from typing import List


EPS = 1e-10
class heat_class(nn.Module):
    def __init__(self, task="wave", device=None) -> None:
        super().__init__()
        self.device = device
        self.task = task

    def twogaussians(self, x, scale=1.0, var=0.8):
        """
        Generate 2 gaussians centered at predefined points.
        Args:
            x (torch.Tensor): Input tensor of shape (N, 2).
            scale (float): Scaling factor for the centers.
            var (float): Variance scaling factor.
        Returns:
            torch.Tensor: Computed gaussian distribution values for input x.
        """
        centers = [(1, 0), (-1, 0)]
        centers = [(scale * x, scale * y) for x, y in centers]
        centers = np.vstack(centers)
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / (torch.max(x0)) * 3 - 1.5
        x1 = x1 / (torch.max(x1)) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)

        out = torch.zeros((x.shape[0])).to(x.device)
        for i in range(centers.shape[0]):
            out += torch.exp(
                (-torch.norm(x_mod - centers[i], dim=1)) * (1 / (var * centers.shape[0])))

        return out + EPS

    def fourgaussians(self, x, scale=1.0, var=0.3):

        centers = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        centers = [(scale * x, scale * y) for x, y in centers]
        centers = np.vstack(centers)
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / (torch.max(x0)) * 3 - 1.5
        x1 = x1 / (torch.max(x1)) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)

        out = torch.zeros((x.shape[0])).to(x.device)
        for i in range(centers.shape[0]):
            out += torch.exp(
                (-torch.norm(x_mod - centers[i], dim=1)) * (1 / (var * centers.shape[0])))

        return out + EPS

    def sixgaussians(self, x, scale=1.0, var=0.01):
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]

        centers = [(scale * x, scale * y) for x, y in centers]
        centers = np.vstack(centers)
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / (torch.max(x0)) * 3 - 1.5
        x1 = x1 / (torch.max(x1)) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)

        out = torch.zeros((x.shape[0])).to(x.device)
        for i in range(centers.shape[0]):
            out += torch.exp(
                (-torch.norm(x_mod - centers[i], dim=1)) * (1 / (var * centers.shape[0])))

        return out + EPS

    def eightgaussians(self, x, scale=1.0, var=0.01):
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]
        centers = np.vstack(centers)
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / (torch.max(x0)) * 3 - 1.5
        x1 = x1 / (torch.max(x1)) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)

        out = torch.zeros((x.shape[0])).to(x.device)
        for i in range(centers.shape[0]):
            out += torch.exp(
                (-torch.norm(x_mod - centers[i], dim=1)) * (1 / (var * centers.shape[0])))

        return out + EPS

    def tengaussians(self, x, scale=1.0, var=0.04):
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), -1. / np.sqrt(2)), (2, 0), (-2, 0)]
        centers = [(scale * x, scale * y) for x, y in centers]
        centers = np.vstack(centers)
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / (torch.max(x0)) * 3 - 1.5
        x1 = x1 / (torch.max(x1)) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)

        out = torch.zeros((x.shape[0])).to(x.device)
        for i in range(centers.shape[0]):
            out += torch.exp(
                (-torch.norm(x_mod - centers[i], dim=1)) * (1 / (var * centers.shape[0])))

        return out + EPS

    def twelvegaussians(self, x, scale=1.0, var=0.03):
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2)), (2, 0), (-2, 0), (0, 2), (0, -2)]
        centers = [(scale * x, scale * y) for x, y in centers]
        centers = np.vstack(centers)
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / (torch.max(x0)) * 3 - 1.5
        x1 = x1 / (torch.max(x1)) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)

        out = torch.zeros((x.shape[0])).to(x.device)
        for i in range(centers.shape[0]):
            out += torch.exp(
                (-torch.norm(x_mod - centers[i], dim=1)) * (1 / (var * centers.shape[0])))

        return out + EPS

    def fourteengaussians(self, x, scale=1.0, var=0.02):
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2)),
            (2, 0), (-2, 0), (0, 2), (0, -2), (2. / np.sqrt(2), 2. / np.sqrt(2)), (-2. / np.sqrt(2), -2. / np.sqrt(2))]

        centers = [(scale * x, scale * y) for x, y in centers]
        centers = np.vstack(centers)
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / (torch.max(x0)) * 3 - 1.5
        x1 = x1 / (torch.max(x1)) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)

        out = torch.zeros((x.shape[0])).to(x.device)
        for i in range(centers.shape[0]):
            out += torch.exp(
                (-torch.norm(x_mod - centers[i], dim=1)) * (1 / (var * centers.shape[0])))

        return out + EPS

    def sixteengaussians(self, x, scale=1.0, var=0.012):
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]
        centers = np.vstack(centers)
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / (torch.max(x0)) * 3 - 1.5
        x1 = x1 / (torch.max(x1)) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)

        out = torch.zeros((x.shape[0])).to(x.device)
        for i in range(centers.shape[0]):
            out += torch.exp(
                (-torch.norm(x_mod - centers[i], dim=1)) * (1 / (var * centers.shape[0])))

        return out + EPS

    def two_t_distributions(self, x, scale=1.0, var=0.05, df=2):
        """
        Generate 2 t-distributions centered at predefined points.
        Args:
            x (torch.Tensor): Input tensor of shape (N, 2).
            scale (float): Scaling factor for the centers.
            var (float): Variance scaling factor.
            df (float): Degrees of freedom for the t-distribution.
        Returns:
            torch.Tensor: Computed t-distribution values for input x.
        """
        centers = [(1, 0), (-1, 0)]
        centers = [(scale * cx, scale * cy) for cx, cy in centers]
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / torch.max(x0) * 3 - 1.5
        x1 = x1 / torch.max(x1) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)
        out = torch.zeros((x.shape[0]), device=x.device)

        for i in range(centers.shape[0]):
            dist = torch.norm(x_mod - centers[i], dim=1) ** 2
            out += (1 + dist / df).pow(-(df + 1) / 2) / torch.sqrt(torch.tensor(df * np.pi, device=x.device))

        out /= centers.shape[0]

        return out + EPS

    def four_t_distributions(self, x, scale=1.0, var=0.05, df=2):
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        centers = [(scale * cx, scale * cy) for cx, cy in centers]
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / torch.max(x0) * 3 - 1.5
        x1 = x1 / torch.max(x1) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)

        out = torch.zeros((x.shape[0]), device=x.device)
        for i in range(centers.shape[0]):
            dist = torch.norm(x_mod - centers[i], dim=1) ** 2
            out += (1 + dist / df).pow(-(df + 1) / 2) / torch.sqrt(torch.tensor(df * np.pi, device=x.device))

        out /= centers.shape[0]

        return out + EPS

    def six_t_distributions(self, x, scale=1.0, var=0.05, df=2):
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), -1. / np.sqrt(2))]

        centers = [(scale * cx, scale * cy) for cx, cy in centers]
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / torch.max(x0) * 3 - 1.5
        x1 = x1 / torch.max(x1) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)

        out = torch.zeros((x.shape[0]), device=x.device)

        for i in range(centers.shape[0]):
            dist = torch.norm(x_mod - centers[i], dim=1) ** 2
            out += (1 + dist / df).pow(-(df + 1) / 2) / torch.sqrt(torch.tensor(df * np.pi, device=x.device))

        out /= centers.shape[0]

        return out + EPS

    def eight_t_distributions(self, x, scale=1.0, var=0.05, df=2):
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), -1. / np.sqrt(2))]

        centers = [(scale * cx, scale * cy) for cx, cy in centers]
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / torch.max(x0) * 3 - 1.5
        x1 = x1 / torch.max(x1) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)
        out = torch.zeros((x.shape[0]), device=x.device)

        for i in range(centers.shape[0]):
            dist = torch.norm(x_mod - centers[i], dim=1) ** 2
            out += (1 + dist / df).pow(-(df + 1) / 2) / torch.sqrt(torch.tensor(df * np.pi, device=x.device))

        out /= centers.shape[0]

        return out + EPS

    def ten_t_distributions(self, x, scale=1.0, var=0.05, df=2):
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2)), (2, 0), (-2, 0)]
        centers = [(scale * cx, scale * cy) for cx, cy in centers]
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / torch.max(x0) * 3 - 1.5
        x1 = x1 / torch.max(x1) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)
        out = torch.zeros((x.shape[0]), device=x.device)

        for i in range(centers.shape[0]):
            dist = torch.norm(x_mod - centers[i], dim=1) ** 2
            out += (1 + dist / df).pow(-(df + 1) / 2) / torch.sqrt(torch.tensor(df * np.pi, device=x.device))

        out /= centers.shape[0]

        return out + EPS

    def twelve_t_distributions(self, x, scale=1.0, var=0.05, df=2):
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2)), (2, 0), (-2, 0), (0, 2),
                   (0, -2)]

        centers = [(scale * cx, scale * cy) for cx, cy in centers]
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / torch.max(x0) * 3 - 1.5
        x1 = x1 / torch.max(x1) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)
        out = torch.zeros((x.shape[0]), device=x.device)

        for i in range(centers.shape[0]):
            dist = torch.norm(x_mod - centers[i], dim=1) ** 2
            out += (1 + dist / df).pow(-(df + 1) / 2) / torch.sqrt(torch.tensor(df * np.pi, device=x.device))

        out /= centers.shape[0]

        return out + EPS

    def fourteen_t_distributions(self, x, scale=1.0, var=0.05, df=2):
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), -1. / np.sqrt(2)), (2, 0), (-2, 0), (0, 2), (0, -2),
                   (2. / np.sqrt(2), 2. / np.sqrt(2)), (-2. / np.sqrt(2), -2. / np.sqrt(2))]

        centers = [(scale * cx, scale * cy) for cx, cy in centers]
        centers = torch.tensor(centers, device=x.device)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / torch.max(x0) * 3 - 1.5
        x1 = x1 / torch.max(x1) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)
        out = torch.zeros((x.shape[0]), device=x.device)

        for i in range(centers.shape[0]):
            dist = torch.norm(x_mod - centers[i], dim=1) ** 2
            out += (1 + dist / df).pow(-(df + 1) / 2) / torch.sqrt(torch.tensor(df * np.pi, device=x.device))

        out /= centers.shape[0]

        return out + EPS

    def sixteen_t_distributions(self, x, scale=1.0, var=0.05, nu=2.0):
        centers = [(1.05, 1.05), (-1.05, 1.05), (1.05, -1.05), (-1.05, -1.05),
                   (0.35, 0.35), (-0.35, 0.35), (-0.35, -0.35), (0.35, -0.35),
                   (1.05, 0.35), (-1.05, 0.35), (-1.05, -0.35), (1.05, -0.35),
                   (0.35, 1.05), (-0.35, 1.05), (-0.35, -1.05), (0.35, -1.05)]
        centers = [(scale * cx, scale * cy) for cx, cy in centers]
        centers = np.vstack(centers)
        centers = torch.tensor(centers, device=x.device, dtype=x.dtype)

        x0 = x[:, 0].reshape(-1, 1)
        x1 = x[:, 1].reshape(-1, 1)
        x0 = x0 / (torch.max(x0)+1e-8) * 3 - 1.5
        x1 = x1 / (torch.max(x1)+1e-8) * 3 - 1.5
        x_mod = torch.cat((x0, x1), dim=1)
        out = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)

        d = 2.0
        alpha = 1.0 / (var * centers.shape[0])

        for i in range(centers.shape[0]):
            dist_sq = torch.clamp(torch.norm(x_mod - centers[i], dim=1) ** 2, max=1e6)
            out += (1.0 + alpha * dist_sq).clamp(min=1e-8).pow(- (nu + d) / 2.0)

        return out + EPS

    def forward(self, x):
        if self.task == "twoG":
            return self.twogaussians(x)
        elif self.task == "fourG":
            return self.fourgaussians(x)
        elif self.task == "sixG":
            return self.sixgaussians(x)
        elif self.task == "eightG":
            return self.eightgaussians(x)
        elif self.task == "tenG":
            return self.tengaussians(x)
        elif self.task == "twelveG":
            return self.twelvegaussians(x)
        elif self.task == "fourteenG":
            return self.fourteengaussians(x)
        elif self.task == "sixteenG":
            return self.sixteengaussians(x)
        elif self.task == "twoT":
            return self.two_t_distributions(x)
        elif self.task == "fourT":
            return self.four_t_distributions(x)
        elif self.task == "sixT":
            return self.six_t_distributions(x)
        elif self.task == "eightT":
            return self.eight_t_distributions(x)
        elif self.task == "tenT":
            return self.ten_t_distributions(x)
        elif self.task == "twelveT":
            return self.twelve_t_distributions(x)
        elif self.task == "fourteenT":
            return self.fourteen_t_distributions(x)
        elif self.task == "sixteenT":
            return self.sixteen_t_distributions(x)
        else:
            raise "Task is not defined"


def get_GB_function(annealing_schedule, GB_values):
    if len(GB_values) == 0:
        raise ValueError("GB_values cannot be empty.")
    GB_values = GB_values[::-1]
    GB_values.append(0)
    GB_values = GB_values[::-1]
    annealing_schedule = annealing_schedule.tolist()[::-1]
    return PchipInterpolator(annealing_schedule, GB_values)



def cal_Barrier(acc: np.ndarray) -> List[float]:
    means_acc = np.mean(acc, axis=1)
    means_rej = 1 - means_acc
    cumulative_sums_of_means = np.cumsum(means_rej).tolist()
    return cumulative_sums_of_means


def compute_emc(samples, M, task):
    """
    Args:
        samples (list or torch.Tensor): Input samples of shape (N, 2).
        M (int): Number of modes (clusters).
    Returns:
        float: The computed EMC value (entropy of the auxiliary distribution).
    """
    eps = 1e-10
    samples_array = np.array(samples)
    if "G" in task:
        probs = p_y_given_xi_G(torch.tensor(samples_array).double(), M)
    elif "T" in task:
        probs = p_y_given_xi_T(torch.tensor(samples_array).double(), M)
    else:
        raise ValueError(f"Unknown task type in args.task: {args.task}")

    log_M = torch.log(torch.tensor(M, dtype=torch.float64))
    entropies = -torch.sum(probs * torch.log(probs + eps) / log_M, dim=1)
    EMC = entropies.mean()

    return EMC.item()


def p_y_given_xi_G(x, M=2):
    means_2 = [(1, 0), (-1, 0)]
    means_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    means_6 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    means_8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    means_10 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    means_12 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2)), (2, 0), (-2, 0), (0, 2), (0, -2)]
    means_14 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2)), (2, 0), (-2, 0), (0, 2), (0, -2), (2. / np.sqrt(2), 2. / np.sqrt(2)), (-2. / np.sqrt(2), -2. / np.sqrt(2))]
    means_16 = [(1.05, 1.05), (-1.05, 1.05), (1.05, -1.05), (-1.05, -1.05),(0.35, 0.35), (-0.35, 0.35), (-0.35, -0.35), (0.35, -0.35), (1.05, 0.35), (-1.05, 0.35), (-1.05, -0.35), (1.05, -0.35),(0.35, 1.05), (-0.35, 1.05), (-0.35, -1.05), (0.35, -1.05)]

    means = np.vstack(eval(f"means_{M}"))
    means = torch.tensor(means, dtype=x.dtype, device=x.device)
    x0 = x[:, 0].reshape(-1, 1)
    x1 = x[:, 1].reshape(-1, 1)
    x0 = x0 / (torch.max(x0)) * 3 - 1.5
    x1 = x1 / (torch.max(x1)) * 3 - 1.5
    x_mod = torch.cat((x0, x1), dim=1)

    distances = torch.cdist(x_mod, means)
    probs = torch.exp(-0.5 * (distances ** 2)* (1 / (0.05)))
    probs /= probs.sum(dim=1, keepdim=True)

    return probs

def p_y_given_xi_T(x, M=2, nu=2):
    means_2 = [(1, 0), (-1, 0)]
    means_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    means_6 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    means_8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    means_10 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
    means_12 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2)), (2, 0), (-2, 0), (0, 2), (0, -2)]
    means_14 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2)), (2, 0), (-2, 0), (0, 2), (0, -2), (2. / np.sqrt(2), 2. / np.sqrt(2)), (-2. / np.sqrt(2), -2. / np.sqrt(2))]
    means_16 = [(1.05, 1.05), (-1.05, 1.05), (1.05, -1.05), (-1.05, -1.05),(0.35, 0.35), (-0.35, 0.35), (-0.35, -0.35), (0.35, -0.35), (1.05, 0.35), (-1.05, 0.35), (-1.05, -0.35), (1.05, -0.35),(0.35, 1.05), (-0.35, 1.05), (-0.35, -1.05), (0.35, -1.05)]

    means = np.vstack(eval(f"means_{M}"))
    means = torch.tensor(means, dtype=x.dtype, device=x.device)

    x0 = x[:, 0].reshape(-1, 1)
    x1 = x[:, 1].reshape(-1, 1)
    x0 = x0 / (torch.max(x0)) * 3 - 1.5
    x1 = x1 / (torch.max(x1)) * 3 - 1.5
    x_mod = torch.cat((x0, x1), dim=1)
    distances = torch.cdist(x_mod, means)
    norm_sq = distances ** 2
    scale = 1 + norm_sq / nu
    probs = torch.pow(scale, -(nu + 2) / 2)
    probs /= probs.sum(dim=1, keepdim=True)

    return probs

############################################ Optimal Tuning ########################################################

def tuning(args):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    max_val = args.max_val
    dim = args.dims
    for sampler_name in args.sampler:
        print("sampler:", sampler_name)
        for task in args.task:
            print("task:", task)
            energy_function = heat_class(task=task, device="cpu")
            sampler = get_sampler(args, sampler_name, dim, max_val, device = device)
            start_coord_y = np.random.randint(0, max_val)
            start_coord_x = np.random.randint(0, max_val)

            start_coord = (start_coord_x, start_coord_y)

            samples, acc_rates, annealing = run_sampler(energy_function=energy_function, batch_size=args.batch_size,
                                             sampling_steps=args.sampling_steps, device=device,
                                             sampler=sampler, sampler_name=sampler_name, start_coord=start_coord,
                                             x_init=None, flags=args)
            print("acc_rates", acc_rates)
            GB = cal_Barrier(acc_rates)
            GB_func = get_GB_function(annealing, GB)  # interpolation
            n_chains = args.chain_number
            GB_1 = GB[0]
            print("GB", GB)
            new_schedule = [0.0]
            for k in range(1, n_chains - 1):
                def fn(x):
                    desired_value = k * GB_1 / (n_chains - 1)
                    return GB_func(x) - desired_value

                new_point = bisect(fn, new_schedule[-1], 1.0)  # bisection
                if new_point >= 1.0:
                    raise ValueError("Encountered value 1.0.")

                new_schedule.append(new_point)

            new_schedule.append(1.0)
            print("old_schedule:", annealing)
            print("new_schedule:", new_schedule[::-1])
            print("Optimal Chain number:", math.ceil((2+3 * GB_1 + math.sqrt(9 * GB_1**2 + 20 * GB_1 + 4)) / 4))

############################################ Sampling ##########################################################

def main(args):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    max_val = args.max_val
    dim = args.dims
    for sampler_name in args.sampler:
        print("sampler:", sampler_name)
        for task in args.task:
            print("task:", task)
            energy_function = heat_class(task=task, device="cpu")
            sampler = get_sampler(args, sampler_name, dim, max_val, device="cpu")
            cur_dir = f"{args.save_dir}/"
            cur_dir += f'{sampler_name}_{args.step1}/'
            print("current dir: ", cur_dir)
            os.makedirs(cur_dir, exist_ok=True)

            samples, probs = calc_probs1(max_val, energy_function, device)
            dist_img = np.zeros((max_val, max_val))
            for i in range(len(samples)):
                coord = samples[i]
                dist_img[coord[0], coord[1]] = probs[i]

            plot_target(dist_img, cur_dir, task)

            start_coord_y = np.random.randint(0, max_val)
            start_coord_x = np.random.randint(0, max_val)

            start_coord = (start_coord_x, start_coord_y)
            est_img = torch.zeros((max_val, max_val))

            samples, acc_rates, annealing = run_sampler(energy_function=energy_function, batch_size=args.batch_size,
                                             sampling_steps=args.sampling_steps, device=device,
                                             sampler=sampler, sampler_name=sampler_name, start_coord=start_coord,
                                             x_init=None, flags=args)
            for i in range(len(samples)):
                coord = samples[i]
                est_img[coord[0], coord[1]] += 1

            target_dist = dist_img
            emp_dist = np.zeros((target_dist.shape))
            burn_in_steps = int(len(samples) * args.burn_in)
            for i in tqdm(range(burn_in_steps, len(samples))):
                idx_ = samples[i]
                emp_dist[idx_[0], idx_[1]] += 1
            emp_dist = emp_dist / np.sum(emp_dist)


            n_component = 8
            emc = compute_emc(samples[:], n_component, "G")  # Gaussian
            # emc = compute_emc(samples[:], n_component, "T")  # T
            mmd = mmd_2d(target_dist, emp_dist, D=1000, sigma=1.0)
            kl = kl_2d(target_dist, emp_dist)

            plot_empirical(est_img, cur_dir, args, sampler_name, task, mmd, kl)

            np.set_printoptions(precision=3, suppress=True, linewidth=400)
            print(f"MMD: {mmd}, KL: {kl}, EMC: {emc}")
            print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--big_step", type=float, default=2.0)
    parser.add_argument("--use_manual_EE", action="store_true")
    parser.add_argument("--big_step_sampling_steps", type=int, default=5)
    parser.add_argument("--small_step", type=float, default=0.2)
    parser.add_argument("--small_bal", type=float, default=0.5)
    parser.add_argument("--big_bal", type=float, default=1.0)

    parser.add_argument("--adapt_every", type=int, default=50)
    parser.add_argument("--burnin_budget", type=int, default=200)
    parser.add_argument("--burnin_adaptive", action="store_true")
    parser.add_argument("--burnin_test_steps", type=int, default=1)
    parser.add_argument("--step_obj", type=str, default="alpha_max")
    parser.add_argument("--burnin_init_bal", type=float, default=0.95)
    parser.add_argument("--a_s_cut", type=float, default=0.5)
    parser.add_argument("--burnin_lr", type=float, default=0.5)
    parser.add_argument("--burnin_error_margin_a_s", type=float, default=0.01)
    parser.add_argument("--burnin_error_margin_hops", type=float, default=5)
    parser.add_argument("--burnin_alphamin_decay", type=float, default=0.9)
    parser.add_argument("--bal_resolution", type=int, default=6)
    parser.add_argument("--burnin_step_obj", type=str, default="alphamax")
    parser.add_argument("--adapt_strat", type=str, default="greedy")
    parser.add_argument("--pair_optim", action="store_true")
    parser.add_argument("--kappa", type=float, default=1)
    parser.add_argument("--continual_adapt_budget", type=int, default=40)

    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--save_dir", type=str, default="./save_dir/")
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--sampling_steps", type=int, default=100)
    parser.add_argument("--chain_number", type=int, default=5)
    parser.add_argument("--burn_in", type=float, default=0.2)
    parser.add_argument("--sampler", type=str, default=["PTDLP"])
    parser.add_argument("--task", type=str,
                        default=['sixteenT'],
                        help="twoT, fourT, sixT, tenT, twelveT, fourteenT, eightT, sixteenT")
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--max_val", type=int, default=101)

    parser.add_argument('--correction', type=float, default=3.0, help="optional, correction in the swap function")
    parser.add_argument("--swap_intensity", type=float, default=0.99, help="optional, swap intensity")
    parser.add_argument('--frozen', type=int, default=0)
    parser.add_argument("--initial_balancing_constant", type=float, default=0.5)
    parser.add_argument("--num_cycles", type=int, default=100)
    parser.add_argument("--use_big", action="store_true")
    parser.add_argument("--min_lr", type=float, default=None)

    args = parser.parse_args()
    set_rand_seeds(args.seed)

    tuning(args)
    #main(args)
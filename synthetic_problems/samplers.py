from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch


def get_sampler(args, sampler_name, dim, max_val, device):
    global sampler
    if sampler_name == "PTDLP":
        sampler = PTDLP(
            dim=dim,
            max_val=max_val,
            n_steps=1,
            correction=args.correction,
            mh=True,
            intensity=args.swap_intensity,
            device=device,
            batch_size=args.batch_size,
            chain_number=args.chain_number
        )
    else:
        raise ValueError(f"Unknown sampler name: {sampler_name}")

    return sampler


def run_sampler(sampler, sampler_name, energy_function, sampling_steps, batch_size, device, start_coord,
                x_init=None, flags=None):
    np.set_printoptions(precision=3, suppress=True, linewidth=400)

    energy_function.one_hot = False

    if x_init is not None:
        x = x_init
    else:
        x = torch.Tensor(start_coord).repeat(batch_size, 1).to(device)
    samples = []
    pg = tqdm(range(sampling_steps))
    for i in pg:
        x = x.to(device)
        energy_function = energy_function.to(device)
        x = sampler.step(x.long().detach(), energy_function).detach()

        samples += list(x.long().detach().cpu().numpy())
        if i % 10 == 0:
            pg.set_description(f"Sampler-{sampler_name}, step_size: {np.round(sampler.step_sizes, 3)}")
    return samples, sampler.s_pairs, sampler.temps


class PTDLP(nn.Module):
    def __init__(self, dim, max_val, chain_number=10, n_steps=1, correction=0,
                 temperature=None, step_size=None, batch_size=2, mh=True, intensity=1., device="cpu"):
        super().__init__()
        self.dim = dim
        self.num_chains = chain_number
        self.n_steps = n_steps
        self.max_val = max_val
        self.correction = correction
        self.batch_size = batch_size
        self.device = device
        self.mh = mh
        self.swap_intensity = intensity

        if temperature is None:
            self.temps = torch.linspace(1.0, 0.0, steps=self.num_chains).to(device)
        else:
            assert len(temperature) == self.num_chains, "Temperature list must match the number of chains"
            self.temps = torch.tensor(temperature, device=device)

        if step_size is None:
            self.step_sizes = ((torch.linspace(.4, .4, steps=self.num_chains) * self.max_val) ** (self.dim ** 2)).to(
                device)
        else:
            assert len(step_size) == self.num_chains, "Step size list must match the number of chains"
            self.step_sizes = torch.tensor(step_size, device=device)

        self.chains = [torch.ones((batch_size, dim), device=device) for _ in range(self.num_chains)]
        self.swap_count = torch.zeros(self.num_chains - 1)

    def get_grad(self, x, model):
        x = x.requires_grad_()
        out = model(x)
        gx = torch.autograd.grad(out.sum(), x, retain_graph=False)[0]

        gx = torch.nan_to_num(gx, nan=0.0, posinf=1e6, neginf=-1e6)
        return gx.detach()

    def _calc_logits(self, x_cur, grad, chain_idx):
        temp = self.temps[chain_idx]
        step_size = self.step_sizes[chain_idx]
        batch_size = x_cur.shape[0]
        disc_values = torch.arange(self.max_val, device=x_cur.device).view(1, 1, -1)
        disc_values = disc_values.repeat(batch_size, self.dim, 1)

        x_expanded = x_cur[:, :, None].repeat(1, 1, self.max_val)
        grad_expanded = grad[:, :, None].repeat(1, 1, self.max_val)
        term1 = temp * grad_expanded * (disc_values - x_expanded) / 2.
        term2 = (disc_values - x_expanded) ** 2 / (2 * step_size)
        return term1 - term2

    def step(self, x, model):
        self.chains[0] = x
        for _ in range(self.n_steps):
            grads = [self.get_grad(chain.float(), model) for chain in self.chains]

            for chain_idx in range(self.num_chains):
                logits = self._calc_logits(self.chains[chain_idx], grads[chain_idx], chain_idx)
                cat_dist = torch.distributions.categorical.Categorical(logits=logits)
                x_delta = cat_dist.sample()

                if self.mh:
                    lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                    grad_delta = self.get_grad(x_delta.float(), model) * self.temps[chain_idx] / 2.
                    logits_reverse = self._calc_logits(x_delta, grad_delta, chain_idx)
                    cat_dist_reverse = torch.distributions.categorical.Categorical(logits=logits_reverse)
                    lp_reverse = torch.sum(cat_dist_reverse.log_prob(self.chains[chain_idx]), dim=1)
                    model_term = model(x_delta).squeeze() - model(self.chains[chain_idx]).squeeze()
                    log_acceptance = model_term + lp_reverse - lp_forward
                    acceptance = (log_acceptance.exp() > torch.rand_like(log_acceptance)).float()
                    self.chains[chain_idx] = x_delta * acceptance[:, None] + self.chains[chain_idx] * (
                                1.0 - acceptance[:, None])
                else:
                    self.chains[chain_idx] = x_delta

            self._swap_chains(model)

        return self.chains[0]

    def _swap_chains(self, model):
        if not hasattr(self, "u"):
            self.u = [None] * self.num_chains
        if not hasattr(self, "s_pairs"):
            self.s_pairs = [[] for _ in range(self.num_chains - 1)]

        for i in range(self.num_chains - 1):
            chain_low = self.chains[i]
            chain_high = self.chains[i + 1]
            temp_low = self.temps[i]
            temp_high = self.temps[i + 1]

            u_low = model(chain_low).detach()
            u_high = model(chain_high).detach()
            if self.u[i] is not None and self.u[i + 1] is not None:
                if torch.isnan(self.u[i]).any() or torch.isnan(self.u[i + 1]).any():
                    s = torch.clamp(torch.exp((temp_high - temp_low) * (u_high - u_low)), max=1.0)
                else:
                    s = torch.clamp(torch.exp((temp_high - temp_low) * (u_high + self.u[i + 1] - u_low - self.u[i])),max=1.0)
            else:
                s = torch.clamp(torch.exp((temp_high - temp_low) * (u_high - u_low)), max=1.0)
            self.u[i] = u_low
            self.u[i + 1] = u_high
            self.s_pairs[i].append(s.detach().mean().item())

            for j in range(u_low.shape[0]):
                if np.random.rand() < self.swap_intensity * s[j].item():
                    chain_low[j], chain_high[j] = chain_high[j].clone(), chain_low[j].clone()
                    self.swap_count[i] += 1

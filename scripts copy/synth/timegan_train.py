# timegan_train.py
# Minimal, readable TimeGAN (PyTorch) for fixed-length multivariate sequences.
# Provides: TimeGAN model + train_timegan() + synthesize().

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

# ---- Modules ----
class Embedder(nn.Module):
    def __init__(self, D: int, H: int):
        super().__init__()
        self.rnn = nn.GRU(D, H, batch_first=True)
    def forward(self, x):  # x: (B,T,D)
        h, _ = self.rnn(x)
        return h  # (B,T,H)

class Recovery(nn.Module):
    def __init__(self, H: int, D: int):
        super().__init__()
        self.fc = nn.Linear(H, D)
    def forward(self, h):  # (B,T,H)
        return self.fc(h)  # (B,T,D)

class Generator(nn.Module):
    def __init__(self, Z: int, H: int):
        super().__init__()
        self.rnn = nn.GRU(Z, H, batch_first=True)
    def forward(self, z):  # (B,T,Z)
        h, _ = self.rnn(z)
        return h  # (B,T,H)

class Supervisor(nn.Module):
    """Optional: maps h_{t-1} -> h_t for stepwise supervision."""
    def __init__(self, H: int):
        super().__init__()
        self.rnn = nn.GRU(H, H, batch_first=True)
    def forward(self, h):         # teacher-forcing on real h
        h_next, _ = self.rnn(h)
        return h_next

class Discriminator(nn.Module):
    def __init__(self, H: int):
        super().__init__()
        self.rnn = nn.GRU(H, H, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*H, 1)
    def forward(self, h):
        u, _ = self.rnn(h)                 # (B,T,2H)
        logits = self.fc(u)                # (B,T,1)
        return logits.squeeze(-1)          # (B,T)

# ---- Losses ----
def reconstruction_loss(x, x_tilde):
    return F.mse_loss(x_tilde, x)

def supervised_step_loss(h_real, h_pred):
    # align one step forward: compare predicted next with real next
    return F.mse_loss(h_pred[:, :-1, :], h_real[:, 1:, :])

def adv_loss_d(logits_real, logits_fake):
    y_real = torch.ones_like(logits_real)
    y_fake = torch.zeros_like(logits_fake)
    return F.binary_cross_entropy_with_logits(logits_real, y_real) + \
           F.binary_cross_entropy_with_logits(logits_fake, y_fake)

def adv_loss_g(logits_fake):
    y = torch.ones_like(logits_fake)
    return F.binary_cross_entropy_with_logits(logits_fake, y)

# ---- Whole model ----
class TimeGAN(nn.Module):
    def __init__(self, D: int, H: int = 128, Z: int = 32):
        super().__init__()
        self.e = Embedder(D,H)
        self.r = Recovery(H,D)
        self.g = Generator(Z,H)
        self.s = Supervisor(H)
        self.d = Discriminator(H)
        self.D, self.H, self.Z = D,H,Z

    @torch.no_grad()
    def synthesize(self, B: int, T: int, device: str):
        z = torch.randn(B, T, self.Z, device=device)
        h_fake = self.g(z)
        x_fake = self.r(h_fake)
        return x_fake

# ---- Train ----
def train_timegan(dataloader_minority: Iterable[torch.Tensor], D: int, T: int, device: str = None,
                  epochs: int = 5000, lr: float = 5e-4, lambda_: float = 1.0, eta: float = 10.0,
                  Z: int = 32, H: int = 128, print_every: int = 200):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeGAN(D,H,Z).to(device)

    opt_er = torch.optim.Adam(list(model.e.parameters())+list(model.r.parameters()), lr=lr)
    opt_g  = torch.optim.Adam(list(model.g.parameters())+list(model.s.parameters()), lr=lr)
    opt_d  = torch.optim.Adam(model.d.parameters(), lr=lr)

    model.train()
    it = 0
    for epoch in range(epochs):
        for x in dataloader_minority:  # x: (B,T,D) scaled to [0,1]
            x = x.to(device)
            B,T,_ = x.shape

            # 1) Train e,r (reconstruction + supervised)
            h_real = model.e(x)
            x_tilde = model.r(h_real)
            Lr = reconstruction_loss(x, x_tilde)
            h_sup = model.s(h_real)
            Ls_er = supervised_step_loss(h_real, h_sup)
            (Lr + lambda_*Ls_er).backward()
            opt_er.step(); opt_er.zero_grad(set_to_none=True)

            # 2) Train d (adversarial) -- detach h_real to avoid backprop to e,r graph
            with torch.no_grad():
                z = torch.randn(B,T,model.Z, device=device)
                h_fake = model.g(z)
            logits_real = model.d(h_real.detach())
            logits_fake = model.d(h_fake)
            Ld = adv_loss_d(logits_real, logits_fake)
            Ld.backward()
            opt_d.step(); opt_d.zero_grad(set_to_none=True)

            # 3) Train g (adversarial + supervised)
            z = torch.randn(B,T,model.Z, device=device)
            h_fake = model.g(z)
            logits_fake = model.d(h_fake)
            Lg_adv = adv_loss_g(logits_fake)
            h_sup = model.s(h_real.detach())
            Ls_g = supervised_step_loss(h_real.detach(), h_sup)
            (eta*Ls_g + Lg_adv).backward()
            opt_g.step(); opt_g.zero_grad(set_to_none=True)

            it += 1
            if print_every and it % print_every == 0:
                print(f"[it={it}] Lr={Lr.item():.4f} Ls={Ls_g.item():.4f} Ld={Ld.item():.4f} Lg={Lg_adv.item():.4f}")

    return model

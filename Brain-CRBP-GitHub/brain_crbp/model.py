"""Brain-CRBP: sparse linear genotype-phenotype projection with two regularisers.

The model keeps one shared SNP projection ``u`` and one projection ``v_t`` per
visit.  For participant ``i`` the clinical score is

    s_i = x_i^T u + sum_t z_i^(t) T v_t

and the training objective (paper Eq. 11) is

    L = lambda_cca  * sum_t ||N(Xu) - N(Z_t v_t)||^2 / n
      + lambda_con  * sum_t  L_rep,t                     (within-subject contrast)
      + lambda_snp  * sum_j  w_j^SNP |u_j|
      + lambda_img  * sum_t sum_j w_j^IMG |v_tj|
      + lambda_reg  * ||d_i^(T) - s_i||_2^2              (sum, not mean)
      + lambda_gwas / 2 * sum_{j in GWAS} (u_j - gwas_scale * beta_j)^2

The entire GWAS term is omitted when no external beta vector is provided.

``L_rep,t`` is the CDReg-style ``-SupCon`` term: every other visit of every
participant sits in the denominator, and only the same participant's visits
with a *different* diagnosis are positive.  Minimising it with a positive
coefficient pushes those pairs apart.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def directional_orders(n_visits: int) -> tuple[tuple[int, ...], ...]:
    """Cyclic candidate order: anchor t is contrasted with t+1, ..., t+T-1."""
    if n_visits < 2:
        raise ValueError("at least two visits are required")
    return tuple(
        tuple((anchor + step) % n_visits for step in range(1, n_visits))
        for anchor in range(n_visits)
    )


def build_pair_masks(dx_visits) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive-pair masks of shape ``(T, N, (T-1)*N)`` and per-block counts.

    Row ``i`` of block ``(anchor, b)`` marks subject ``i`` at candidate visit
    ``b`` when that visit carries a different diagnosis than the anchor visit.
    All ``(T-1)*N`` candidates stay in the denominator, so stable participants
    are still available as negatives.
    """
    dx_visits = torch.as_tensor(dx_visits, dtype=torch.long)
    if dx_visits.ndim != 2:
        raise ValueError("dx_visits must have shape (N, T)")
    n_subjects, n_visits = dx_visits.shape
    orders = directional_orders(n_visits)
    masks = torch.zeros(
        (n_visits, n_subjects, (n_visits - 1) * n_subjects), dtype=torch.float32
    )
    counts = torch.zeros((n_visits, n_visits - 1), dtype=torch.long)
    for anchor, candidates in enumerate(orders):
        for block, candidate in enumerate(candidates):
            changed = torch.where(dx_visits[:, anchor] != dx_visits[:, candidate])[0]
            counts[anchor, block] = changed.numel()
            masks[anchor, changed, block * n_subjects + changed] = 1.0
    return masks, counts


class BrainCRBP(nn.Module):
    """Shared SNP projection + one imaging/protein projection per visit."""

    def __init__(
        self,
        n_snp: int,
        n_img: int,
        n_visits: int = 4,
        contrast_dim: int = 16,
        temperature: float = 0.07,
        snp_penalty: torch.Tensor | None = None,
        img_penalty: torch.Tensor | None = None,
        eps: float = 1e-8,
        gwas_beta: torch.Tensor | None = None,
        gwas_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if n_visits < 2:
            raise ValueError("at least two visits are required")
        self.n_visits = int(n_visits)
        self.temperature = float(temperature)
        self.eps = float(eps)

        self.gene_weight = nn.Parameter(torch.randn(n_snp, 1))
        self.image_weights = nn.Parameter(torch.randn(n_visits, n_img, 1))
        self.image_con_head = nn.Sequential(
            nn.Linear(n_img, contrast_dim, bias=False),
            nn.ReLU(inplace=True),
        )

        # Biological priors are fixed buffers: they weight the penalty but are
        # never updated by the optimiser.
        self.register_buffer("gene_penalty_weights", self._penalties(snp_penalty, n_snp, "SNP"))
        image = self._penalties(img_penalty, n_img, "image")
        self.register_buffer("image_penalty_weights", image.view(1, n_img, 1).repeat(n_visits, 1, 1))

        # None buffers are absent from state_dict, preserving old checkpoints.
        # Missing SNPs are excluded from the full anchor, including its L2 part.
        if gwas_beta is None:
            if gwas_mask is not None:
                raise ValueError("GWAS mask requires an external beta vector")
            beta, mask = None, None
        else:
            beta = torch.as_tensor(gwas_beta, dtype=torch.float32)
            if tuple(beta.shape) not in ((n_snp,), (n_snp, 1)):
                raise ValueError(f"GWAS beta must have shape ({n_snp},) or ({n_snp}, 1)")
            if not bool(torch.isfinite(beta).all()):
                raise ValueError("GWAS beta coefficients must be finite")
            beta = beta.reshape(n_snp, 1).detach().clone()
            mask = torch.ones_like(beta, dtype=torch.bool)
            if gwas_mask is not None:
                supplied = torch.as_tensor(gwas_mask, device=beta.device)
                if tuple(supplied.shape) not in ((n_snp,), (n_snp, 1)):
                    raise ValueError("GWAS mask must have the same feature dimension as beta")
                if not bool(((supplied == 0) | (supplied == 1)).all()):
                    raise ValueError("GWAS mask must contain boolean or 0/1 values")
                mask = supplied.reshape(n_snp, 1).bool().detach().clone()
            if not bool(mask.any()):
                raise ValueError("GWAS beta must cover at least one SNP")
        self.register_buffer("gwas_beta", beta)
        self.register_buffer("gwas_mask", mask)

    @staticmethod
    def _penalties(values, width: int, label: str) -> torch.Tensor:
        values = torch.ones(width, 1) if values is None else torch.as_tensor(values, dtype=torch.float32)
        if tuple(values.shape) not in ((width,), (width, 1)):
            raise ValueError(f"{label} penalty weights must have shape ({width},) or ({width}, 1)")
        values = values.reshape(width, 1)
        if not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
            raise ValueError(f"{label} penalty weights must be finite and positive")
        return values.detach().clone()

    def _cca_normalize(self, values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(dim=0, keepdim=True)
        return centered / centered.norm(dim=0, keepdim=True).clamp_min(self.eps)

    def cca_residual(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = self._cca_normalize(left)
        right = self._cca_normalize(right)
        return (left - right).pow(2).sum(dim=1).mean()

    def contrastive(self, anchors, candidates, positive_mask) -> torch.Tensor:
        positive_mask = positive_mask.to(device=anchors.device, dtype=anchors.dtype)
        positive_counts = positive_mask.sum(dim=1)
        valid = positive_counts > 0
        if not bool(valid.any()):
            return (anchors.sum() + candidates.sum()) * 0.0
        anchors = F.normalize(anchors, dim=1, eps=self.eps)
        candidates = F.normalize(candidates, dim=1, eps=self.eps)
        logits = anchors.mm(candidates.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        mean_positive = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)
        return mean_positive[valid].mean()

    def forward(
        self,
        batch,
        lambda_cca: float = 1.0,
        lambda_con: float = 0.0,
        lambda_snp: float = 0.0,
        lambda_img: float = 0.0,
        lambda_reg: float = 0.0,
        lambda_gwas: float = 1.0,
        gwas_scale: float = 1.0,
    ):
        if not math.isfinite(lambda_reg) or lambda_reg < 0:
            raise ValueError("lambda_reg must be finite and nonnegative")
        if not math.isfinite(lambda_gwas) or lambda_gwas < 0:
            raise ValueError("lambda_gwas must be finite and nonnegative")
        if not math.isfinite(gwas_scale) or gwas_scale < 0:
            raise ValueError("gwas_scale must be finite and nonnegative")
        snp = batch["snp"]
        img = batch["img_visits"]
        masks = batch["pair_masks"]
        n_subjects = snp.shape[0]
        if snp.ndim != 2 or n_subjects == 0:
            raise ValueError("snp must be a non-empty 2-D tensor")
        if img.ndim != 3 or img.shape[1] != self.n_visits:
            raise ValueError(f"img_visits must have shape (N, {self.n_visits}, n_img)")
        if img.shape[0] != n_subjects or img.shape[2] != self.image_weights.shape[1]:
            raise ValueError("SNP and image subject/feature dimensions do not match the model")
        if tuple(masks.shape) != (self.n_visits, n_subjects, (self.n_visits - 1) * n_subjects):
            raise ValueError("pair_masks do not match the batch")

        z_gene = snp.mm(self.gene_weight)
        z_img = torch.einsum("nvd,vdk->nvk", img, self.image_weights)
        score = z_gene + z_img.sum(dim=1)

        dx = batch.get("dx_visits")
        if dx is None:
            if lambda_reg > 0:
                raise ValueError("dx_visits is required when lambda_reg is positive")
            loss_reg = score.new_zeros(())
        else:
            if tuple(dx.shape) != (n_subjects, self.n_visits):
                raise ValueError("dx_visits must have shape (N, T)")
            target = dx[:, -1:].to(device=score.device, dtype=score.dtype)
            if not bool(torch.isfinite(target).all()):
                raise ValueError("last-visit diagnosis target must be finite")
            loss_reg = (target - score).square().sum()

        z_con = self.image_con_head(img * self.image_weights.squeeze(-1).unsqueeze(0))

        cca_terms = [self.cca_residual(z_gene, z_img[:, visit]) for visit in range(self.n_visits)]
        con_terms = []
        for anchor, candidates in enumerate(directional_orders(self.n_visits)):
            stack = torch.cat([z_con[:, visit] for visit in candidates], dim=0)
            con_terms.append(self.contrastive(z_con[:, anchor], stack, masks[anchor]))

        loss_cca = torch.stack(cca_terms).sum()
        loss_con = torch.stack(con_terms).sum()
        loss_snp = (self.gene_penalty_weights * self.gene_weight.abs()).sum()
        img_l1_terms = (self.image_penalty_weights * self.image_weights.abs()).sum(dim=(1, 2))
        loss_img = img_l1_terms.sum()

        total = lambda_cca * loss_cca + lambda_con * loss_con
        total = total + lambda_snp * loss_snp + lambda_img * loss_img
        if lambda_reg > 0:
            total = total + lambda_reg * loss_reg

        loss_gwas = self.gene_weight.new_zeros(())
        if self.gwas_beta is not None:
            residual = (self.gene_weight[self.gwas_mask]
                        - gwas_scale * self.gwas_beta[self.gwas_mask])
            loss_gwas = 0.5 * residual.square().sum()
            if lambda_gwas > 0:
                total = total + lambda_gwas * loss_gwas

        out = {
            "loss_total": total,
            "loss_cca": loss_cca,
            "loss_con": loss_con,
            "loss_snp_l1": loss_snp,
            "loss_img_l1": loss_img,
            "loss_reg": loss_reg,
            "loss_gwas": loss_gwas,
            "score": score,
            "z_gene": z_gene,
        }
        for visit in range(self.n_visits):
            out[f"loss_cca_t{visit + 1}"] = cca_terms[visit]
            out[f"loss_con_t{visit + 1}"] = con_terms[visit]
            out[f"z_img_t{visit + 1}"] = z_img[:, visit]
        return out

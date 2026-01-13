from typing import Optional
import torch

class _Loss(torch.nn.Module):
    """Base loss with reduction and optional padding index masking.

    Parameters
    ----------
    reduction: str
        One of {"none", "mean", "sum", "custom"}. Custom reduces by
        summing per-row then averaging across rows.
    pad_ind: Optional[int]
        Optional padding index to be masked (set to zero) in 2D losses.
    """

    def __init__(self, reduction: str = "mean", pad_ind: Optional[int] = None):
        super(_Loss, self).__init__()
        self.reduction = reduction
        self.pad_ind = pad_ind

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply configured reduction to a loss tensor."""
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "custom":
            return loss.sum(dim=1).mean()
        else:
            return loss.sum()

    def _mask_at_pad(self, loss: torch.Tensor) -> torch.Tensor:
        """Mask the loss at padding index by setting it to zero."""
        if self.pad_ind is not None:
            loss[:, self.pad_ind] = 0.0
        return loss

    def _mask(self, loss: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply a boolean mask to the loss (False entries become zero)."""
        if mask is not None:
            loss = loss.masked_fill(~mask, 0.0)
        return loss

class RegLoss(_Loss):
    """Simple margin-based regularization over positive and negative pairs."""

    def __init__(self, reduction: str = "mean", margin: float = 0.1, k: int = 10):
        super(RegLoss, self).__init__(reduction=reduction)
        self.margin = margin
        self.k = k

    def forward(
        self,
        sim_i: torch.Tensor,
        sim_f: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            indices = torch.multinomial(target, 1, replacement=False)

        # get positive loss
        sim_p_i = sim_i.gather(1, indices)
        sim_p_f = sim_f.gather(1, indices)
        loss_p = torch.max(
            torch.zeros_like(sim_p_i), sim_p_i - sim_p_f + self.margin
        ).flatten()

        # get negative loss
        similarities = torch.where(target == 0, sim_f, torch.full_like(sim_f, -50))
        k = min(self.k, similarities.size(1))
        _, indices = torch.topk(similarities, largest=True, dim=1, k=k)
        sim_n_f = sim_f.gather(1, indices)
        sim_n_i = sim_i.gather(1, indices)
        loss_n = torch.max(
            torch.zeros_like(sim_n_f), sim_n_f - sim_n_i + self.margin
        ).flatten()

        # Handle cases where there is no signal to regularize
        if loss_p.sum() == 0:
            loss_p = 0
        else:
            loss_p = loss_p.sum() / (loss_p > 0).sum()

        if loss_n.sum() == 0:
            loss_n = 0
        else:
            loss_n = loss_n.sum() / (loss_n > 0).sum()
        return (loss_p + loss_n) / 2

class ATripletMarginLossOHNMDM(_Loss):
    r"""Triplet Margin Loss with Online Hard Negative Mining.

    Applies loss using top-k hardest negatives per-row with optional
    softmax weighting and a dynamic margin bounded between
    `margin_min` and `margin_max`.
    """

    def __init__(
        self,
        args=None,
        reduction="mean",
        margin_min=0.1,
        margin_max=0.3,
        k=3,
        apply_softmax=False,
        tau=0.1,
        num_violators=False,
        select_fixed_pos=False,
        train_loader=None,
    ):
        super(ATripletMarginLossOHNMDM, self).__init__(reduction=reduction)
        self.margin_min = margin_min
        self.margin_max = margin_max
        self.k = k
        self.tau = tau
        self.num_violators = num_violators
        self.apply_softmax = apply_softmax
        self.select_fixed_pos = select_fixed_pos
        self.args = args
        self.train_loader = train_loader
        
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet-style margin loss with OHNM.

        Parameters
        ----------
        input: torch.Tensor
            Similarity matrix of shape (batch_size, output_size).
        target: torch.Tensor
            Binary ground-truth matrix of shape (batch_size, output_size).
        mask: Optional[torch.Tensor]
            Boolean mask for entries to consider.

        Returns
        -------
        torch.Tensor
            Reduced loss according to `self.reduction`.
        """
        
        if self.args.fixed_pos == True:
            sim_p = torch.diagonal(input).view(-1, 1)
        else:
            with torch.no_grad():
                indices_p = torch.multinomial(target, 1, replacement=False)
            # get similarity of positives
            sim_p = input.gather(1, indices_p)

            
        similarities = torch.where(target == 0, input, torch.full_like(input, -50))
        k = min(self.k, similarities.size(1))
        _, indices = torch.topk(similarities, largest=True, dim=1, k=k)
        sim_n = input.gather(1, indices)
        
        if self.margin_min == self.margin_max:
            d_margin = self.margin_min
        else:
            d_margin = torch.clamp(
                input=torch.abs(sim_p - sim_n).detach(),
                min=self.margin_min,
                max=self.margin_max,
            )
        loss = torch.max(torch.zeros_like(sim_p), sim_n - sim_p + d_margin)
        
        
        if self.apply_softmax:
            sim_n[loss == 0] = -50
            prob = torch.softmax(sim_n / self.tau, dim=1)
            loss = loss * prob
            
        reduced_loss = self._reduce(loss)

        if self.num_violators:
            nnz = torch.sum((loss > 0), axis=1).float().mean()
            return reduced_loss, nnz
        else:
            return reduced_loss

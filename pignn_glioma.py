"""
Physics-Informed Graph Neural Network (PI-GNN) for glioma growth prediction.

Core idea
---------
Each graph node = a supervoxel / atlas parcel of brain tissue, carrying a
scalar tumor cell density c_i in [0, 1]. Edge weights are precomputed from
DTI (anisotropic diffusion tensor), NOT learned from scratch -- they encode
the finite-volume discretization of div(D grad c). A small MLP learns a
patient/tissue-specific *correction* to those physics-derived weights and to
the per-node proliferation rate rho_i. Message passing = one explicit time
step of the Fisher-KPP reaction-diffusion PDE:

    dc/dt = div(D grad c) + rho * c * (1 - c)

This file is a SKELETON: data loading, DTI edge-weight precomputation, and
FEM synthetic-data generation are stubbed out with clear TODOs. Fill those
in with your actual preprocessing pipeline (nibabel / dipy for DTI, your
FEM solver for synthetic trajectories).

Dependencies: torch, torch_geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ---------------------------------------------------------------------------
# 1. Physics-informed message passing layer
# ---------------------------------------------------------------------------
class ReactionDiffusionStep(MessagePassing):
    """
    One explicit time step of:
        c_i(t+1) = c_i(t) + dt * [ sum_j w_ij_learned * (c_j - c_i)
                                    + rho_i_learned * c_i * (1 - c_i) ]

    w_ij (base) is the physics-derived diffusion conductance from DTI.
    w_ij_learned = w_ij * sigmoid(MLP_theta(edge_features))
    rho_i_learned = softplus(MLP_phi(node_features))
    """

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, hidden_dim: int = 32,
                 rho_per_node: bool = True):
        # node_dim=0 (not the MessagePassing default of -2): c is a plain 1D
        # (N,) density tensor, not the (N, F) feature convention the default
        # assumes, so -2 is out of range on c. node_dim=0 also correctly
        # covers edge_attr's (E, F) axis, so it works for both.
        super().__init__(aggr="add", node_dim=0)  # sum over neighbors, matches PDE discretization

        # Edge correction MLP: takes [edge_features, c_i, c_j] -> scalar gate in (0,1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # rho_per_node=True (default): reproduces every existing call site exactly
        # -- a node-feature-conditioned MLP predicts a per-node, per-patient
        # proliferation rate (patient adaptivity is already baked into the
        # zero-shot forward pass, before any per-patient rho_scale fit happens).
        #
        # rho_per_node=False: ABLATION, added to causally test a hypothesis about
        # why per-patient rho_scale personalization (inverse_fit_patient()) gives
        # PI-GNN only a small gain (+0.008 Dice on RHUH-GBM CV) while the exact
        # same single-scalar personalization gives the FEM baseline a much larger
        # one (+0.076 Dice) -- see TODO_Nature_readiness.md, Tier 2. FEMBaseline's
        # rho is a single global constant with NO per-node/per-patient
        # conditioning at all, unlike rho_mlp here. This flag makes PI-GNN's
        # zero-shot rho behave the same way (one global learnable scalar, no
        # node-feature conditioning), so it can train and then run the exact
        # same evaluate_personalized_pi() personalization experiment on it: if
        # the hypothesis is right, THIS model's personalization gain should jump
        # toward FEM's ~+0.076, since a genuinely patient-blind zero-shot rho
        # leaves much more room for a global per-patient correction to fill in,
        # matching FEM's situation. If personalization gain stays small even
        # here, the "rho_mlp already captures patient adaptivity" hypothesis is
        # wrong and the real bottleneck lies elsewhere.
        self.rho_per_node = rho_per_node
        if rho_per_node:
            self.rho_mlp = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.rho_const = nn.Parameter(torch.tensor(0.0))  # softplus(0)=~0.69, a reasonable init

        self.dt = 0.1  # discrete time step; treat as hyperparameter / could be learned
        self._last_gate = None  # stashed by message() each forward call; read by physics_prior_loss below

    def forward(self, c, node_features, edge_index, edge_attr, rho_scale=1.0):
        """
        c:              (N,) current tumor density per node
        node_features:  (N, node_feat_dim) static features (tissue type, distance to
                         resection cavity, baseline segmentation, etc.)
        edge_index:     (2, E) graph connectivity
        edge_attr:      (E, edge_feat_dim) includes the physics-derived base
                         conductance w_ij as one of the features -- see
                         build_edge_features() below
        rho_scale:      scalar (float or 0-dim/1-dim tensor, e.g. an
                         nn.Parameter being optimized by inverse_fit_patient())
                         multiplying the learned proliferation rate. Default
                         1.0 reproduces the previous behavior exactly for
                         every existing call site (train_multi_patient(),
                         evaluate_model_on_patients(), etc.) that doesn't
                         pass it.
        """
        diffusion_update = self.propagate(edge_index, c=c, edge_attr=edge_attr)

        if self.rho_per_node:
            rho_base = F.softplus(self.rho_mlp(node_features)).squeeze(-1)  # (N,), per-node/patient
        else:
            # ablation: same scalar for every node/patient, matching FEMBaseline's
            # rho -- see rho_per_node's docstring note above for why this exists.
            rho_base = F.softplus(self.rho_const).expand(c.shape[0])  # (N,), one value broadcast
        rho = rho_base * rho_scale  # (N,)
        reaction_update = rho * c * (1 - c)

        c_next = c + self.dt * (diffusion_update + reaction_update)
        return c_next.clamp(0.0, 1.0), rho  # clamp keeps density physically valid

    def message(self, c_i, c_j, edge_attr):
        # edge_attr[:, 0] is assumed to be the precomputed physics base weight w_ij
        w_base = edge_attr[:, 0]
        gate = torch.sigmoid(
            self.edge_mlp(torch.cat([edge_attr, c_i.unsqueeze(-1), c_j.unsqueeze(-1)], dim=-1))
        ).squeeze(-1)
        self._last_gate = gate  # exposed for physics_prior_loss -- see note there
        w_learned = w_base * gate
        return w_learned * (c_j - c_i)


# ---------------------------------------------------------------------------
# 2. Full model: unroll the PDE step over T timesteps
# ---------------------------------------------------------------------------
class GliomaGrowthPIGNN(nn.Module):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, hidden_dim: int = 32,
                 rho_per_node: bool = True):
        """
        rho_per_node: passed straight through to ReactionDiffusionStep -- see its
        docstring. Default True reproduces every existing call site exactly;
        False is the constant-rho ablation for the personalization-bottleneck
        hypothesis test (TODO_Nature_readiness.md, Tier 2).
        """
        super().__init__()
        self.step = ReactionDiffusionStep(node_feat_dim, edge_feat_dim, hidden_dim,
                                           rho_per_node=rho_per_node)

    def forward(self, c0, node_features, edge_index, edge_attr, n_steps: int, rho_scale=1.0):
        """
        Simulate forward from baseline density c0 for n_steps.
        Returns the full trajectory (for the PDE residual loss) and the
        final state (for the data-fit loss against the follow-up scan).

        rho_scale: passed straight through to each ReactionDiffusionStep
        call; see its docstring. Default 1.0 keeps every existing call site
        unchanged.
        """
        c = c0
        trajectory = [c]
        rhos = []
        for _ in range(n_steps):
            c, rho = self.step(c, node_features, edge_index, edge_attr, rho_scale=rho_scale)
            trajectory.append(c)
            rhos.append(rho)
        return torch.stack(trajectory, dim=0), torch.stack(rhos, dim=0)  # (T+1, N), (T, N)


# ---------------------------------------------------------------------------
# 3. Losses
# ---------------------------------------------------------------------------
def dice_loss(pred, target, eps: float = 1e-6):
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + eps) / (pred.sum() + target.sum() + eps)


def physics_prior_loss(trajectory, node_features, edge_index, edge_attr, step_module, dt):
    """
    Physics-prior regularizer on the LEARNED corrections (replaces an earlier
    "pde_residual_loss" that was mathematically vacuous -- see note below for
    why, kept here so the failure mode isn't silently lost to history).

    Why the original formulation was a no-op:
    GliomaGrowthPIGNN.forward() generates `trajectory` by unrolling
    step_module itself: c_next = c + dt * (diffusion(c) + reaction(c)), and
    trajectory[t+1] IS c_next. Recomputing step_module(trajectory[t]) and
    comparing the result to trajectory[t+1] therefore just repeats the exact
    arithmetic that produced trajectory[t+1] in the first place -- the
    "residual" is 0 to floating-point precision for every timestep and every
    set of weights, trained or not (confirmed empirically on real RHUH-GBM
    and LUMIERE graphs: pde=0.0 exactly). It contributed no gradient signal.
    A genuine PDE-residual term only makes sense when the trajectory being
    checked is NOT itself generated by literally executing the same operator
    being checked against -- e.g. free/latent intermediate states in a
    collocation-style PINN. This architecture instead mechanistically
    simulates the trajectory, so that formulation doesn't apply here.

    What this regularizes instead:
    the two corrections ReactionDiffusionStep learns on top of the
    DTI-derived physics -- the per-edge diffusion gate
    sigmoid(edge_mlp(...)) in (0, 1) (1 = "trust the physics-derived
    conductance as-is"), and the per-node proliferation rate rho from
    rho_mlp (0 = "no learned growth beyond diffusion"). Penalizing
    (gate - 1)^2 and rho^2 anchors the simulation to the DTI physics prior
    unless L_data (the fit to the real follow-up scan) pulls it away --
    which is the actual behavior wanted here: physically-plausible dynamics
    between the two sparse real observations, not a literal PDE residual.
    """
    gate_penalties = []
    rho_penalties = []
    for t in range(trajectory.shape[0] - 1):
        c_t = trajectory[t]
        _, rho = step_module(c_t, node_features, edge_index, edge_attr)
        gate = step_module._last_gate  # set as a side effect of message() inside the call above
        gate_penalties.append(F.mse_loss(gate, torch.ones_like(gate)))
        rho_penalties.append((rho ** 2).mean())
    return torch.stack(gate_penalties).mean() + torch.stack(rho_penalties).mean()


def boundary_loss(c_final, boundary_mask):
    """Zero-flux / zero-density boundary at skull and ventricle nodes."""
    return (c_final[boundary_mask] ** 2).mean()


def total_loss(trajectory, c_observed_final, node_features, edge_index, edge_attr,
                step_module, boundary_mask, lambdas=(1.0, 0.5, 0.1)):
    lam_data, lam_prior, lam_bc = lambdas
    c_final = trajectory[-1]

    L_data = dice_loss(c_final, c_observed_final)
    L_prior = physics_prior_loss(trajectory, node_features, edge_index, edge_attr,
                                  step_module, dt=step_module.dt)
    L_bc = boundary_loss(c_final, boundary_mask)

    return lam_data * L_data + lam_prior * L_prior + lam_bc * L_bc, {
        "data": L_data.item(), "physics_prior": L_prior.item(), "bc": L_bc.item()
    }


# ---------------------------------------------------------------------------
# 4. Edge feature construction (physics precomputation) -- TODO: fill in
# ---------------------------------------------------------------------------
def build_edge_features(node_centroids, diffusion_tensors, edge_index, shared_boundary_area):
    """
    Compute the finite-volume conductance w_ij = (A_ij / h_ij^2) * n_ij^T D_ij n_ij
    for every edge, from DTI-derived diffusion tensors.

    node_centroids:     (N, 3) node positions in physical space (mm)
    diffusion_tensors:  (N, 3, 3) DTI tensor per node (interpolate to edge midpoint)
    edge_index:         (2, E)
    shared_boundary_area: (E,) precomputed shared face area between adjacent supervoxels

    Returns: edge_attr (E, edge_feat_dim), where column 0 = w_ij (base physics weight)
    """
    src, dst = edge_index
    diff = node_centroids[dst] - node_centroids[src]
    h_ij = diff.norm(dim=-1) + 1e-8
    n_ij = diff / h_ij.unsqueeze(-1)

    D_mid = 0.5 * (diffusion_tensors[src] + diffusion_tensors[dst])  # (E, 3, 3)
    Dn = torch.einsum("eij,ej->ei", D_mid, n_ij)
    n_D_n = torch.einsum("ei,ei->e", n_ij, Dn)

    w_ij = (shared_boundary_area / h_ij.pow(2)) * n_D_n

    # TODO: append additional edge features here (e.g. tissue-type mismatch
    # flag, distance itself) -- keep w_ij as column 0 since message() expects it
    edge_attr = torch.stack([w_ij, h_ij], dim=-1)
    return edge_attr


# ---------------------------------------------------------------------------
# 5. Training loop skeleton
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        trajectory, _ = model(
            c0=batch.c0,
            node_features=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            n_steps=batch.n_steps[0].item(),  # assumes uniform n_steps per batch
        )

        loss, components = total_loss(
            trajectory, batch.c_followup, batch.x, batch.edge_index, batch.edge_attr,
            model.step, batch.boundary_mask,
        )
        loss.backward()
        optimizer.step()
        total += loss.item()

    return total / len(loader)


def inverse_fit_patient(model, x, edge_index, edge_attr, c0, c_followup, boundary_mask,
                         n_steps: int, n_iters: int = 200, lr: float = 1e-2,
                         lambdas=(1.0, 0.0, 0.0)):
    """
    Freeze theta/phi (already trained). Fit a single per-patient scalar
    (rho_scale, multiplying the learned proliferation rate) by backprop
    against that patient's own baseline + follow-up scan. This replaces slow
    iterative FEM fitting with a single fast gradient-based fit.

    Previously a placeholder: rho_scale was optimized against a loss that
    never actually depended on it (model(...) was called without rho_scale,
    so every gradient step no-opped and the routine could not produce a
    usable per-patient fit). Fixed now that ReactionDiffusionStep.forward()
    and GliomaGrowthPIGNN.forward() both accept and apply rho_scale (see
    their docstrings) -- rho_scale is passed through on every call below, so
    the loss genuinely depends on it and the Adam step has a real gradient
    to follow.

    lambdas defaults to (1.0, 0.0, 0.0), NOT total_loss()'s own default of
    (1.0, 0.5, 0.1): those pretraining-scale weights on lam_prior/lam_bc were
    tuned for the full theta/phi training run, not this single-scalar
    200-iteration fit, and physics_prior_loss() computes its rho penalty by
    calling step_module(c_t, ...) WITHOUT rho_scale -- so that penalty is
    evaluated at the module's hardcoded default (rho_scale=1.0), never at the
    candidate value actually being optimized here. It cannot regularize the
    parameter being fit; it can only add an unrelated, constant-scale term to
    the objective and dilute the gradient signal coming from L_data. Zeroing
    lam_prior/lam_bc makes this a pure data-fit (Dice) objective against the
    patient's own follow-up scan, which is the correct objective for a single
    free scalar parameter. Callers that want the old (confounded) weighting
    back can still pass lambdas=(1.0, 0.5, 0.1) explicitly.

    Takes the same (x, edge_index, edge_attr, c0, c_followup, boundary_mask)
    tuple that graph_dict_to_tensors() returns elsewhere in this pipeline,
    rather than a torch_geometric Data object with those as attributes --
    matches how every other function in this file/notebook is actually
    called on real patient graphs.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    rho_scale = torch.nn.Parameter(torch.tensor(1.0, device=c0.device))
    optimizer = torch.optim.Adam([rho_scale], lr=lr)

    for _ in range(n_iters):
        optimizer.zero_grad()
        trajectory, _ = model(c0, x, edge_index, edge_attr, n_steps=n_steps, rho_scale=rho_scale)
        loss, _ = total_loss(trajectory, c_followup, x, edge_index, edge_attr,
                              model.step, boundary_mask, lambdas=lambdas)
        loss.backward()
        optimizer.step()

    return rho_scale.item()


# ---------------------------------------------------------------------------
# TODO checklist for your actual implementation:
# 1. build_edge_features(): load DTI via dipy/nibabel, compute per-supervoxel
#    mean tensor, wire in real shared_boundary_area from your parcellation.
# 2. Dataset class: wrap RHUH-GBM / LUMIERE / UCSF-PDGM into torch_geometric
#    Data objects with fields: x, edge_index, edge_attr, c0, c_followup,
#    boundary_mask, n_steps.
# 3. FEM synthetic data generator: separate script, not shown here, to
#    pretrain before fine-tuning on real longitudinal scans.
# (inverse_fit_patient()'s rho_scale threading is fixed above, and now
#  exercised in the Section 8 CV loop via evaluate_personalized_pi(): each
#  fold's frozen model_pi gets a per-patient rho_scale fit on its own
#  held-out validation patients, compared against zero-shot PI-GNN and a
#  per-patient FEM fit on the same patients.)
# 4. [NEW] rho_per_node=False ablation on GliomaGrowthPIGNN/ReactionDiffusionStep:
#    to run the causal test described in TODO_Nature_readiness.md (Tier 2,
#    personalization bottleneck), add a cell to the notebook, right after the
#    existing Section 8 CV loop, e.g.:
#
#      model_rho_const, _ = train_multi_patient(
#          rhuh_graphs, train_ids, n_epochs=N_EPOCHS, n_steps=N_STEPS,
#          physics_informed=True, seed=fold_i,
#      )  # NOTE: train_multi_patient() will need a rho_per_node passthrough
#         # kwarg added to its GliomaGrowthPIGNN(...) construction call, since
#         # it isn't threaded there yet -- a small edit, not shown here since
#         # train_multi_patient() isn't defined in this file (it's notebook-only).
#      rows_rho_const_personalized = evaluate_personalized_pi(
#          model_rho_const, rhuh_graphs, val_ids, n_steps=N_STEPS,
#      )
#      print("constant-rho ablation, personalized:",
#            summarize_rows(rows_rho_const_personalized,
#                            keys=("dice", "coverage", "rmse", "fit_time_ms")))
#
#    Compare this ablation's zero-shot-vs-personalized Dice gain against the
#    existing +0.008 (PI-GNN) and +0.076 (FEM) numbers already in Table 2 /
#    Section 3.5. This has NOT been run (no torch/torch_geometric in this dev
#    sandbox) -- it's ready for the next real Kaggle run, not yet executed.
# ---------------------------------------------------------------------------

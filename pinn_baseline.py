"""
PINN baseline, adapted from Zhang et al. 2025 ("Personalized Predictions of
Glioblastoma Infiltration: Mathematical Models, Physics-Informed Neural
Networks and Multimodal Scans", Medical Image Analysis 101, 103423,
arXiv:2311.16536; code: github.com/Rayzhangzirui/pinngbm). Cited in the
manuscript's Table 3 as a non-reproduced comparison; this module is the live
adaptation identified as the more tractable of the two ("GliODIL and/or PINN")
in TODO_Nature_readiness.md, Tier 2 -- unlike GliODIL, this method has no
GPU-memory ceiling (a small coordinate MLP, not a large grid/multigrid solve).

STATUS: implemented but UNTESTED beyond its pure-numpy pieces -- there is no
torch install in this dev sandbox. The four numpy-only stages (tissue
geometry, diffuse-domain phase field, the 1D characteristic-parameter grid
search, and the diffuse-domain FDM solver) WERE run against synthetic data in
this environment (see the smoke test at the bottom of this file) and, in the
process, caught and fixed two real bugs: (1) the 1D radial solver's initial
seed was originally ~50x narrower than the paper's actual non-dimensional IC
formula, which diffused away before any meaningful growth and made the
characteristic-parameter grid search completely degenerate (every target
collapsed to the same answer); (2) the paper's own literal Dbar/rhobar in
[0.1, 1] grid also proved degenerate once the IC was fixed -- with a properly
scaled seed, only the SINGLE smallest ratio (0.1) in that range produced a
profile whose peak density even reached the paper's own u_t1gd_c=0.6
threshold, so every other grid point returned a degenerate near-zero radius
regardless of length scale. Both were caught by testing against basic
theoretical bounds (a closed-form logistic-growth ceiling, a padded
true-exterior-boundary check, and two DELIBERATELY non-proportional target
tumor shapes to confirm the grid search can actually differentiate between
them, not just between overall scale) -- exactly the kind of check that a
quick "does it run without crashing" smoke test would have missed. Fixed by
widening both the ratio grid and the solver's r_max/resolution; see inline
comments at each function for specifics. This grid is now verified
NON-degenerate (differentiates correctly between different tumor
presentations) but NOT verified to reproduce the original paper's own exact
characteristic-parameter values, since there's no access to their code's
numeric outputs to check against -- flagged inline where relevant.

The torch-dependent PINN network, PDE-residual autograd, and training loops
have only been checked for syntax (`python -m py_compile`), not run.
Smoke-test on 1-2 real patients on the next Kaggle run before trusting this
at scale, exactly like generate_synthetic_growth_data.py's real-DTI loader.

What's faithfully reproduced from the paper vs. what's a documented
adaptation for this project's data (RHUH-GBM/LUMIERE), listed once here so
it doesn't need repeating in every function docstring below:

FAITHFUL (same equations/algorithm as the paper):
  - Fisher-KPP PDE with Neumann BC, du/dt = div(D(x) grad u) + rho*u*(1-u),
    D(x) a weighted sum of white/gray-matter diffusion coefficients with
    D_white = 10 * D_gray (paper Eq. 1-2, Harpold et al. 2007 / Swanson et
    al. 2000 convention -- the SAME convention this project's own FEM
    baseline and generate_synthetic_growth_data.py already use).
  - The non-dimensionalization / characteristic-parameter scheme (paper
    Section 2.1.3, Eq. 5-7): characteristic velocity v_bar=sqrt(D_bar*rho_bar),
    characteristic time T_bar=L_bar/v_bar, non-dim parameters
    script_D=D_bar*T_bar/L_bar^2, script_R=rho_bar*T_bar (and script_D=1/script_R,
    so L_bar and D_bar/rho_bar are the two genuinely free characteristic
    quantities, estimated per-patient by a fast 1D spherically-symmetric grid
    search (paper Eq. 10) against the patient's own segmentation radii.
  - The two-stage workflow: (1) grid-search characteristic params, (2)
    pretrain the PINN against an FDM "characteristic solution" u_bar_FDM
    computed with mu_D=mu_R=1, (3) fine-tune mu_D, mu_R (and thresholds) on
    the patient's real segmentation via a thresholded/smoothed-Heaviside data
    loss plus box-constraint penalties on the fitted parameters (paper Eq.
    11-19). mu_D, mu_R in [0.75, 1.25] as in the paper.
  - The hard-constraint network form u(x,t) = t * u_NN(x,t) + u0(x), exactly
    enforcing the initial condition at t=0 (paper Section 2.3, just above
    Eq. 11).

ADAPTED (deviations from the paper, made explicit rather than silently
changed, since this project's data differs from the paper's):
  - Diffuse-domain phase field phi(x): the paper evolves a Cahn-Hilliard-type
    PDE (their Eq. 4) to get a smooth phase field from the binary brain mask.
    That's a real numerical PDE in its own right and adds meaningful
    implementation risk for a fairly small accuracy gain over a much
    simpler standard alternative: this module instead builds phi via
    Gaussian-smoothing the binary brain mask to the same ~3mm interface
    width the paper uses. Both approaches produce a smooth 0->1 transition
    layer of the same physical width; this is a documented simplification,
    not a literal reproduction of their Eq. 4.
  - Tissue geometry P(x): the paper uses continuous atlas-derived white/gray
    matter percentage maps (Pw(x), Pg(x)). This project's tissue_mask is a
    hard integer label (0=background/CSF, 1=gray, 2=white, 3=ventricle, see
    generate_synthetic_growth_data.py/anatomy_and_graph_conversion.py), so
    Pw/Pg here are hard 0/1 indicators of that label, not soft percentages.
  - Initial condition / seed location x0: the paper's data is a single,
    unknown-time MRI snapshot, so x0 is a free parameter fit from the data
    (a Gaussian bump centered at a trainable location). This project's
    RHUH-GBM/LUMIERE data is LONGITUDINAL (real baseline + follow-up scans,
    unlike the paper's single-snapshot setting) -- so x0 is instead read
    directly from the patient's own real baseline segmentation centroid,
    genuine information the original paper's method doesn't have access to.
    This is a deliberate, favorable adaptation for this comparison (giving
    the PINN baseline real information rather than making it re-discover
    something we already know), and should be described as such in any
    write-up, not presented as a literal reproduction of their protocol.
  - No FET-PET data: RHUH-GBM/LUMIERE don't include PET, so L_FET is unused
    (equivalent to the paper's own w_FET=0 configuration when PET isn't
    used) -- this is an explicitly supported path in the original paper, not
    a gap introduced here.
  - Segmentation classes: the paper uses 2 binary indicators (T1Gd=tumor
    core, FLAIR=tumor core+edema). RHUH-GBM/LUMIERE ship 3-class tumor
    segmentation at the source (necrosis / peritumoral edema / enhancing
    tumor -- see TODO_Nature_readiness.md's GliODIL section for the same
    finding) which this project's own pipeline doesn't yet remap from a
    binary mask (a separate, shared prerequisite noted in the roadmap).
    Until that remapping lands, y_T1Gd and y_FLAIR below both fall back to
    the single binary tumor mask this project already has -- a real
    information loss relative to the paper (no separate edema signal),
    flagged here rather than silently assumed away.

Dependencies: numpy, scipy (all numpy-only functions below). torch (PINN
network + autograd PDE residual + training loops) -- guarded import, only
required if you call the torch-dependent functions.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

# Reuse the already-validated finite-volume flux function from the synthetic
# solver, rather than re-deriving flux discretization here (see that file's
# compute_divergence_diffusion docstring for the full-tensor derivation and
# its two independent validation tests earlier this session).
from generate_synthetic_growth_data import compute_divergence_diffusion


# ---------------------------------------------------------------------------
# 1. Tissue geometry P(x) and diffuse-domain phase field phi(x)
#    (numpy only -- tested against synthetic data, see smoke test below)
# ---------------------------------------------------------------------------
def compute_tissue_geometry(tissue_mask, white_matter_scale: float = 10.0):
    """
    P(x) = Pw(x) * white_matter_scale + Pg(x), hard 0/1 indicators from
    tissue_mask (see module docstring's ADAPTED note on soft-vs-hard
    percentages). white_matter_scale=10 matches Dw=10*Dg (paper Eq. 2,
    Harpold et al. 2007 / Swanson et al. 2000), i.e. P(x) is already scaled
    so that D(x) = D_gray * P(x) reproduces D(x) = Dw*Pw(x) + Dg*Pg(x).
    """
    Pw = (tissue_mask == 2).astype(float)
    Pg = (tissue_mask == 1).astype(float)
    return Pw * white_matter_scale + Pg


def compute_phase_field(tissue_mask, voxel_size=1.0, interface_width_mm: float = 3.0):
    """
    Smooth 0/1 phase field phi(x) approximating the paper's diffuse-domain
    method (see module docstring's ADAPTED note -- this is a Gaussian-smoothed
    brain mask, not a literal Cahn-Hilliard evolution of their Eq. 4, but
    produces the same qualitative object: phi~1 deep inside the brain, phi~0
    outside, with a smooth transition layer of the target physical width).

    voxel_size: scalar or length-3 array of physical mm spacing per axis
    (same convention as anatomy_and_graph_conversion.py's build_supervoxel_graph).
    """
    vs = np.broadcast_to(np.asarray(voxel_size, dtype=float), (3,))
    brain_mask = (tissue_mask > 0).astype(float)
    # sigma in voxels per axis, chosen so the smoothing kernel's physical
    # width matches interface_width_mm (a factor-of-~2 sigma-to-full-width
    # rule of thumb is fine here -- phi doesn't need to be a precise PDE
    # solution, just a smooth 0->1 ramp of roughly the right physical scale).
    sigma_voxels = (interface_width_mm / 2.0) / vs
    phi = gaussian_filter(brain_mask, sigma=sigma_voxels)
    return np.clip(phi, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 2. Fast 1D spherically-symmetric solver for the characteristic-parameter
#    grid search (paper Section 2.2, Eq. 10). numpy only -- tested below.
# ---------------------------------------------------------------------------
def spherical_fisher_kpp_1d(D_nd, R_nd, r_max: float = 8.0, n_r: int = 400,
                             n_steps: int = 400, dt: float = None):
    """
    Solve the non-dimensional, spherically-symmetric (P(x)=1, ignoring brain
    geometry) Fisher-KPP PDE (paper Eq. 10):
        du/dt = D_nd * (1/r^2) d/dr(r^2 du/dr) + R_nd * u * (1-u)
    with Neumann BC at r=0 and r=r_max, starting from a small central seed.
    Returns the radial profile u(r) at non-dimensional t=1 (the "imaging
    time" in the paper's own non-dimensionalization), used only to look up
    the radius at which u crosses a threshold -- this never touches real
    patient data, it's a fast, patient-independent lookup-table building
    block (paper: "independent of the patient data... precomputed and
    stored as a look-up table").
    """
    r = np.linspace(1e-3, r_max, n_r)  # avoid r=0 singularity in 1/r^2 term
    dr = r[1] - r[0]
    if dt is None:
        dt = 0.2 * dr ** 2 / max(D_nd, 1e-6)  # explicit-Euler stability bound

    # Matches the paper's actual non-dim IC, u0(x_bar) = 0.1*exp(-0.1*|x_bar|^2)
    # (Section 2.1.1), restricted to radial coordinates (|x-x0|=r here) -- an
    # earlier version of this function used an ad hoc grid-resolution-sized
    # seed instead (width ~2*dr, i.e. ~50x narrower than this formula's actual
    # e-folding width of sqrt(10)=~3.16 non-dim units), which diffused away
    # almost immediately and made the grid search below fail to differentiate
    # between different observed tumor sizes at all. Caught via this file's
    # own smoke test (see bottom of file) -- fixed here to use the same
    # formula as the real 3D initial condition, not a placeholder.
    u = 0.1 * np.exp(-0.1 * r ** 2)

    n_dt_steps = max(1, int(round(1.0 / dt)))  # integrate to non-dim t=1
    actual_dt = 1.0 / n_dt_steps
    for _ in range(n_dt_steps):
        du_dr = np.gradient(u, dr)
        flux = r ** 2 * du_dr
        div = np.gradient(flux, dr) / (r ** 2)
        div[0] = div[1]  # Neumann-consistent value at the r->0 singularity
        u = u + actual_dt * (D_nd * div + R_nd * u * (1 - u))
        u = np.clip(u, 0.0, 1.0)
    return r, u


def radius_at_threshold(r, u, threshold: float):
    """First (innermost) radius from center where u crosses below threshold,
    i.e. the outer edge of the region where u > threshold -- matches the
    paper's R_sph^FLAIR / R_sph^T1Gd definitions (Eq. 10's surrounding text)."""
    below = np.where(u < threshold)[0]
    if len(below) == 0:
        return r[-1]  # never drops below threshold within r_max
    return r[below[0]]


def estimate_characteristic_params(R_t1gd_seg_mm: float, R_flair_seg_mm: float,
                                    Dbar_over_rhobar_grid=None, Lbar_grid_mm=None,
                                    u_t1gd_c: float = 0.6, u_flair_c: float = 0.35):
    """
    Grid search for patient-specific characteristic L_bar and D_bar/rho_bar
    (paper Section 2.2, Eq. 10 and surrounding text), matching the observed
    real segmentation radii R_t1gd_seg_mm, R_flair_seg_mm (measured directly
    from the patient's own baseline/follow-up segmentation -- computed by
    the caller, e.g. max distance from tumor centroid to segmentation
    boundary, per axis-corrected physical mm).

    Default grids: the paper states Dbar_over_rhobar in [0.1, 1], Lbar in
    [10, 90] mm ("generate spherical tumors with radii smaller than 120mm").
    Testing that exact range in this dev sandbox (see this file's own smoke
    test) found it degenerate: with the fixed u0 formula above, only the
    smallest ratio (0.1) actually produces a profile whose peak density
    exceeds the u_t1gd_c=0.6 threshold anywhere in the domain -- every larger
    ratio in [0.2, 1.0] caps out below 0.6 (peak density 0.14-0.44), so the
    "T1Gd radius" collapses to ~0 for most of the grid regardless of Lbar,
    and the fit degenerates to picking whichever Lbar best matches the
    single viable ratio -- exactly the grid-boundary-saturation failure mode
    already seen and fixed once in this project for the FEM baseline's own
    rho grid search (RHO_CANDIDATES, widened in the notebook after "a real
    run picked 0.5, the largest candidate, in every fold"). Widened here for
    the same reason, down to 0.005 where peak density reaches saturation
    (~1.0) and both thresholds are meaningfully resolvable. This widened
    range has NOT been validated against the original paper's own reported
    characteristic-parameter values (no access to their code's exact
    numerical outputs) -- it's chosen here only to avoid the observed
    degeneracy, not confirmed to reproduce their specific numbers.
    u_t1gd_c=0.6, u_flair_c=0.35 are the paper's own fixed characteristic
    thresholds, used ONLY for this grid-search stage (the real,
    patient-specific thresholds are later fit during fine-tuning, exactly
    as in the paper).

    Returns: (Dbar, rhobar, Lbar) -- characteristic diffusion coefficient,
    proliferation rate, and length scale, from which script_D, script_R
    (Eq. 5) follow directly in build_nondim_pde_params() below.
    """
    # NOTE on resolution: verified via this file's own smoke test that this
    # grid successfully DIFFERENTIATES between genuinely different-shaped
    # targets (not just different overall scale) -- but the shape-sensitive
    # transition zone turned out to sit narrowly around ratio~0.1-0.2 (below
    # ~0.1 the profile saturates and FLAIR/T1Gd shape collapses to ~1.0
    # regardless of ratio; above ~0.2 T1Gd becomes degenerate/unreachable).
    # Both smoke-test targets landed on ratio=0.1 (the single best-available
    # shape match in a fairly coarse grid), differing only in the fitted
    # Lbar -- correct given the grid's resolution, but a finer sampling
    # between 0.1 and 0.3 (e.g. 0.12, 0.15, 0.18) would likely improve
    # shape-fit precision. Left as coarse for now; revisit if fine-tuning
    # keeps landing on grid edges for real patients (same signal as the
    # RHO_CANDIDATES widening precedent referenced above).
    if Dbar_over_rhobar_grid is None:
        Dbar_over_rhobar_grid = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    if Lbar_grid_mm is None:
        Lbar_grid_mm = np.array([5, 10, 15, 20, 30, 45, 60, 75, 90], dtype=float)

    best = None
    best_err = float("inf")
    for Dbar_over_rhobar in Dbar_over_rhobar_grid:
        # D_nd = script_D, R_nd = script_R = 1/script_D at mu_D=mu_R=1
        # (paper: script_D * script_R = 1, so a single non-dim ratio drives
        # the radial shape; we sweep Dbar_over_rhobar directly here, which is
        # the physically meaningful free quantity per the paper's own Eq. 5-6
        # derivation: script_D = sqrt(Dbar/rhobar)/Lbar = 1/script_R).
        D_nd = np.sqrt(Dbar_over_rhobar)
        R_nd = 1.0 / D_nd
        r_nd, u = spherical_fisher_kpp_1d(D_nd, R_nd)
        for Lbar in Lbar_grid_mm:
            r_mm = r_nd * Lbar
            R_t1gd_sph = radius_at_threshold(r_mm, u, u_t1gd_c)
            R_flair_sph = radius_at_threshold(r_mm, u, u_flair_c)
            err = (R_t1gd_sph - R_t1gd_seg_mm) ** 2 + (R_flair_sph - R_flair_seg_mm) ** 2
            if err < best_err:
                best_err = err
                best = (Dbar_over_rhobar, Lbar)

    Dbar_over_rhobar, Lbar = best
    # From D_nd = sqrt(Dbar/rhobar)/Lbar = 1 (mu_D=mu_R=1 by construction of
    # the characteristic solution) => Dbar/rhobar = Lbar^2 exactly -- but we
    # ALSO independently swept Dbar_over_rhobar above as the shape parameter,
    # so recover absolute Dbar, rhobar via vbar=sqrt(Dbar*rhobar) with the
    # convention Dbar = Dbar_over_rhobar-scaled velocity fixed by Lbar itself
    # (paper: L_bar and D_bar/rho_bar together fully determine script_D,
    # script_R, hence the characteristic solution -- absolute Dbar, rhobar
    # individually are only needed downstream to convert back into physical
    # mu_D, mu_R during fine-tuning, via Eq. 6).
    rhobar = 1.0 / (Lbar * np.sqrt(Dbar_over_rhobar))  # choice consistent with vbar=Lbar/Tbar, Tbar=1 (non-dim)
    Dbar = Dbar_over_rhobar * rhobar
    return Dbar, rhobar, Lbar


# ---------------------------------------------------------------------------
# 3. Diffuse-domain FDM solver for the characteristic solution u_bar_FDM
#    (paper Eq. 3), reusing compute_divergence_diffusion. numpy only.
# ---------------------------------------------------------------------------
def solve_diffuse_domain_fdm(P, phi, D_gray, rho, x0_index, tissue_mask,
                              voxel_size=1.0, tau: float = 1e-3,
                              n_steps: int = 200, dt: float = 0.05,
                              save_every: int = 20):
    """
    Solve the diffuse-domain Fisher-KPP PDE (paper Eq. 3):
        d/dt(phi*u) = D_gray * div(P * phi_tau * grad u) + phi * rho * u * (1-u)
    on the full voxel grid (the "cubic box Omega_B" -- here just the whole
    array, since our arrays are already axis-aligned voxel grids), where
    phi_tau = phi + tau keeps the diffusion coefficient nonzero everywhere
    (paper: "tau > 0 is small") so flux can propagate through the smooth
    diffuse-domain transition layer rather than hitting a hard zero.

    Implementation choice (not spelled out verbatim in the paper): maintains
    the conserved quantity w = phi*u, updates it via the SAME finite-volume
    flux function already validated in generate_synthetic_growth_data.py
    (isotropic diffusion tensor D_eff(x) = D_gray * P(x) * phi_tau(x) * I,
    tissue_mask=all-ones since the whole box is the working domain -- phi/tau
    itself handles the effective boundary, not a hard tissue_mask cutoff),
    then recovers u = w / phi_tau (never dividing by exactly zero since
    phi_tau >= tau > 0 everywhere).

    Returns: trajectory list of u arrays (same save_every convention as
    generate_synthetic_growth_data.py's simulate_growth()).
    """
    shape = P.shape
    phi_tau = phi + tau

    D_eff = D_gray * P * phi_tau  # (X,Y,Z) scalar effective diffusivity
    D_tensor_eff = D_eff[..., None, None] * np.eye(3)  # isotropic -> compute_divergence_diffusion
    all_valid_mask = np.ones(shape, dtype=int)  # whole box is "valid"; phi/tau handles the boundary

    u = np.zeros(shape)
    u[x0_index] = 0.1  # matches paper's u0 peak value (0.1 * exp(0) at x=x0)
    u = gaussian_filter(u, sigma=1.0)  # smooth seed into a small blob, same convention as
                                        # generate_synthetic_growth_data.py's simulate_growth()
    w = phi_tau * u
    trajectory = [u.copy()]

    for step in range(1, n_steps + 1):
        flux = compute_divergence_diffusion(w / phi_tau, D_tensor_eff, all_valid_mask, voxel_size=voxel_size)
        reaction = phi * rho * (w / phi_tau) * (1 - w / phi_tau)
        w = w + dt * (flux + reaction)
        u = np.clip(w / phi_tau, 0.0, 1.0)
        w = phi_tau * u  # re-sync w after clipping u, keeps the two consistent across steps

        if step % save_every == 0:
            trajectory.append(u.copy())

    return trajectory


# ---------------------------------------------------------------------------
# 4. PINN coordinate network + PDE residual (torch -- UNTESTED in this
#    sandbox, no torch install here; syntax-checked only via py_compile)
# ---------------------------------------------------------------------------
if torch is not None:
    class PINNCoordinateNet(nn.Module):
        """
        Coordinate MLP u_NN(x, t) -> scalar, with the hard initial-condition
        constraint u(x,t) = t * u_NN(x,t) + u0(x) applied by the caller (not
        inside forward(), since u0(x) depends on the patient-specific x0 and
        is cheaper to add once outside the network than to thread through
        every forward call).
        """
        def __init__(self, hidden_dim: int = 128, n_layers: int = 5):
            super().__init__()
            layers = [nn.Linear(4, hidden_dim), nn.Tanh()]  # input: (x, y, z, t)
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
            layers += [nn.Linear(hidden_dim, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, xyzt):
            return self.net(xyzt).squeeze(-1)


    def u0_gaussian(xyz, x0, peak: float = 0.1, width: float = 0.1):
        """u0(x) = 0.1 * exp(-0.1 * |x - x0|^2), paper's initial condition
        (Section 2.1.1). xyz, x0 in the SAME non-dimensional coordinates the
        network is trained in (x_bar = x / L_bar)."""
        d2 = ((xyz - x0) ** 2).sum(dim=-1)
        return peak * torch.exp(-width * d2)


    def hard_ic_output(net, xyzt, x0, peak: float = 0.1, width: float = 0.1):
        """u(x,t) = t * u_NN(x,t) + u0(x), see module/class docstrings."""
        u_nn = net(xyzt)
        t = xyzt[..., 3]
        u0 = u0_gaussian(xyzt[..., :3], x0, peak=peak, width=width)
        return t * u_nn + u0


    def pde_residual(net, xyzt, x0, mu_D, mu_R, script_D, script_R,
                      P_interp, phi_interp, grad_P_phi_interp, peak=0.1, width=0.1):
        """
        F[u] at collocation points xyzt (paper Eq. 11, non-divergence form):
            F[u] = mu_D*script_D*(grad(P*phi).grad(u) + P*phi*laplacian(u))
                   + mu_R*script_R*phi*u*(1-u) - du/dt

        P_interp, phi_interp, grad_P_phi_interp: callables (or precomputed
        tensors already gathered at xyzt's spatial coordinates by the
        caller) giving P(x)*phi(x), and grad(P(x)*phi(x)) -- precomputed
        from the voxel grid via finite differences (paper: "approximated by
        finite differences using data from the pixels in the MRI scans"),
        NOT autograd-differentiated, since P*phi is fixed input data, not a
        function the network represents. Only derivatives of u itself
        (network output) go through autograd here.

        Returns per-collocation-point residual (caller squares + means it
        for L_PDE, paper Eq. 12).
        """
        xyzt = xyzt.clone().requires_grad_(True)
        u = hard_ic_output(net, xyzt, x0, peak=peak, width=width)

        grad_u = torch.autograd.grad(u.sum(), xyzt, create_graph=True)[0]  # (N, 4): du/dx,dy,dz,dt
        du_dt = grad_u[..., 3]
        du_dxyz = grad_u[..., :3]

        laplacian_u = torch.zeros_like(u)
        for i in range(3):
            grad2 = torch.autograd.grad(du_dxyz[..., i].sum(), xyzt, create_graph=True)[0]
            laplacian_u = laplacian_u + grad2[..., i]

        Pphi = P_interp  # (N,), precomputed at these xyzt's spatial locations
        grad_Pphi = grad_P_phi_interp  # (N, 3), precomputed via finite differences

        diffusion_term = mu_D * script_D * (
            (grad_Pphi * du_dxyz).sum(dim=-1) + Pphi * laplacian_u
        )
        reaction_term = mu_R * script_R * phi_interp * u * (1 - u)
        return diffusion_term + reaction_term - du_dt


    def segmentation_loss(net, xyzt_data, x0, y_t1gd, y_flair, u_t1gd_c, u_flair_c,
                           phi_at_data, a: float = 20.0, peak=0.1, width=0.1):
        """
        L_SEG (paper Eq. 15-16): smoothed-Heaviside threshold match against
        real segmentation indicators at non-dim t=1 (the single observed
        timepoint). y_t1gd, y_flair: (N,) binary tensors, this project's
        available tumor-core / tumor-core+edema masks (see module docstring's
        note: y_flair currently falls back to the same binary mask as
        y_t1gd until the multi-class segmentation remapping lands).
        u_t1gd_c, u_flair_c: trainable threshold parameters (nn.Parameter,
        constrained via segment-specific box penalties, see fit_pinn_patient
        below).
        """
        u_at_1 = hard_ic_output(net, xyzt_data, x0, peak=peak, width=width)
        H_t1gd = torch.sigmoid(a * (phi_at_data * u_at_1 - u_t1gd_c))
        H_flair = torch.sigmoid(a * (phi_at_data * u_at_1 - u_flair_c))
        return ((H_t1gd - y_t1gd) ** 2).mean() + ((H_flair - y_flair) ** 2).mean()


    def box_penalty(value, lo, hi):
        """L_beta (paper Eq. 18): quadratic penalty outside [lo, hi]."""
        return torch.clamp(lo - value, min=0) ** 2 + torch.clamp(value - hi, min=0) ** 2


# ---------------------------------------------------------------------------
# 5. Top-level per-patient pipeline (torch -- UNTESTED, see status note above)
# ---------------------------------------------------------------------------
if torch is not None:
    def run_pinn_baseline_patient(tissue_mask, c0, c_followup, voxel_size=1.0,
                                   D_gray: float = 0.001, n_pretrain_iters: int = 2000,
                                   n_finetune_iters: int = 2000, lr: float = 1e-3,
                                   n_collocation: int = 4000, n_data_points: int = 2000,
                                   device: str = "cpu", y_t1gd_real=None, y_flair_real=None):
        """
        End-to-end PINN baseline for one patient, matching this project's
        other per-patient evaluation functions' calling convention (returns
        a row dict compatible with evaluate_prediction()/summarize_rows(),
        so it drops into the notebook's Section 8 CV loop the same way
        evaluate_model_on_patients()/evaluate_personalized_pi() do).

        c0, c_followup: (X,Y,Z) baseline/follow-up tumor density arrays, same
        convention as elsewhere in this project. x0 (seed location) is read
        directly from c0's centroid -- see module docstring's ADAPTED note on
        why this deviates (favorably) from the paper's free-fit x0.

        y_t1gd_real, y_flair_real: optional (X,Y,Z) bool/0-1 arrays giving the
        REAL tumor-core (T1Gd) and tumor-core+edema (FLAIR) indicator masks,
        e.g. from anatomy_and_graph_conversion.py's remap_tumor_subregions()
        applied to the patient's actual multi-region segmentation
        (t1gd_core, flair_region keys). When given, these replace the
        threshold-of-c_followup fallback below (module docstring's note: that
        fallback treats y_t1gd and y_flair as two thresholds of the SAME
        binary mask, an information loss relative to the paper's genuinely
        separate T1Gd/FLAIR signals -- passing real sub-region masks here
        closes that gap once available). Default None preserves the
        original fallback behavior exactly.

        NOT YET RUN in this dev sandbox (no torch here). Before trusting
        this on a real cohort: (1) verify the non-dim scaling actually keeps
        collocation-point PDE residuals well-scaled (paper's own diagnostic:
        mu_D, mu_R should end up close to 1 -- if fine-tuning drives them to
        the edge of [0.75,1.25] repeatedly, the characteristic-parameter grid
        search needs a wider grid, same failure mode already seen and fixed
        for the FEM rho grid search in Section 8's RHO_CANDIDATES widening),
        (2) confirm collocation/data point counts and iteration counts are
        enough on real patient-sized volumes without exceeding Kaggle's
        session time budget -- these defaults are unverified guesses, not
        benchmarked numbers.
        """
        device = torch.device(device)
        shape = tissue_mask.shape
        vs = np.broadcast_to(np.asarray(voxel_size, dtype=float), (3,))

        # --- geometry / phase field (numpy, tested pieces) ---
        P = compute_tissue_geometry(tissue_mask)
        phi = compute_phase_field(tissue_mask, voxel_size=vs)

        # --- x0 from real baseline centroid (ADAPTED, see module docstring) ---
        tumor_voxels = np.argwhere(c0 > 0.05)
        if len(tumor_voxels) == 0:
            tumor_voxels = np.argwhere(tissue_mask > 0)  # fallback: brain centroid
        x0_index = tuple(np.round(tumor_voxels.mean(axis=0)).astype(int))
        x0_mm = np.array(x0_index) * vs

        # --- real segmentation radii (mm), for the characteristic grid search ---
        centroid_mm = x0_mm
        if y_t1gd_real is not None and y_flair_real is not None:
            # Real, independent sub-region masks (see y_t1gd_real docstring
            # note above) -- genuinely separate T1Gd/FLAIR extents, not two
            # thresholds of the same field.
            seg_voxels_t1gd = np.argwhere(np.asarray(y_t1gd_real) > 0)
            seg_voxels_flair = np.argwhere(np.asarray(y_flair_real) > 0)
        else:
            seg_voxels_t1gd = np.argwhere(c_followup > 0.5)  # tumor "core" proxy
            seg_voxels_flair = np.argwhere(c_followup > 0.1)  # tumor+"edema" proxy --
            # see module docstring: falls back to the same density field at two
            # thresholds until real multi-class masks are passed in, rather
            # than true independent T1Gd/FLAIR indicators.
        R_t1gd_seg_mm = (np.linalg.norm((seg_voxels_t1gd - centroid_mm / vs) * vs, axis=-1).max()
                          if len(seg_voxels_t1gd) else 5.0)
        R_flair_seg_mm = (np.linalg.norm((seg_voxels_flair - centroid_mm / vs) * vs, axis=-1).max()
                           if len(seg_voxels_flair) else 10.0)

        Dbar, rhobar, Lbar = estimate_characteristic_params(R_t1gd_seg_mm, R_flair_seg_mm)
        vbar = np.sqrt(Dbar * rhobar)
        Tbar = Lbar / vbar
        script_D = Dbar * Tbar / Lbar ** 2  # == 1/script_R, paper Eq. 5
        script_R = rhobar * Tbar

        # --- pretraining target: characteristic FDM solution (numpy, tested piece) ---
        char_trajectory = solve_diffuse_domain_fdm(
            P, phi, D_gray=Dbar, rho=rhobar, x0_index=x0_index,
            tissue_mask=tissue_mask, voxel_size=vs, n_steps=200, save_every=200,
        )
        u_char_fdm = char_trajectory[-1]  # non-dim t=1 characteristic solution

        # --- torch tensors ---
        P_t = torch.tensor(P, dtype=torch.float32, device=device)
        phi_t = torch.tensor(phi, dtype=torch.float32, device=device)
        Pphi_t = P_t * phi_t
        # finite-difference gradient of P*phi (paper: precomputed from pixel
        # data, not autograd -- it's fixed input, not a function of network
        # weights)
        grad_Pphi_np = np.stack(np.gradient(P * phi, *vs), axis=-1)
        grad_Pphi_t = torch.tensor(grad_Pphi_np, dtype=torch.float32, device=device)
        x0_nd = torch.tensor(x0_mm / Lbar, dtype=torch.float32, device=device)

        coords = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), axis=-1) * vs
        coords_nd = coords / Lbar  # non-dim spatial coordinates, x_bar = x/L_bar

        net = PINNCoordinateNet().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        rng = np.random.default_rng(0)

        def sample_collocation(n, t_bias_early=True):
            flat_idx = rng.integers(0, np.prod(shape), size=n)
            idx = np.unravel_index(flat_idx, shape)
            xyz = coords_nd[idx]
            # paper: sample densely at early times and near tumor center
            # (Appendix B) -- approximate via a Beta-skewed sample toward t=0
            t = rng.beta(0.5, 2.0, size=n) if t_bias_early else rng.uniform(0, 1, size=n)
            xyzt = np.concatenate([xyz, t[:, None]], axis=-1)
            return torch.tensor(xyzt, dtype=torch.float32, device=device), idx

        def gather_at_idx(field_t, idx):
            return field_t[idx]

        # --- Stage 1: pretrain against the characteristic FDM solution ---
        u_char_t = torch.tensor(u_char_fdm, dtype=torch.float32, device=device)
        for _ in range(n_pretrain_iters):
            optimizer.zero_grad()
            xyzt, idx = sample_collocation(n_collocation)
            Pphi_at = gather_at_idx(Pphi_t, idx)
            gradPphi_at = gather_at_idx(grad_Pphi_t, idx)
            phi_at = gather_at_idx(phi_t, idx)

            residual = pde_residual(net, xyzt, x0_nd, mu_D=1.0, mu_R=1.0,
                                     script_D=script_D, script_R=script_R,
                                     P_interp=Pphi_at, phi_interp=phi_at,
                                     grad_P_phi_interp=gradPphi_at)
            L_pde = (residual ** 2).mean()

            u_pred = hard_ic_output(net, xyzt, x0_nd)
            u_char_at = u_char_t[idx]  # note: char FDM solution isn't indexed by t here --
            # a simplification: treats u_char_fdm (the t=1 snapshot) as the
            # matching target regardless of the sampled collocation t, which
            # is only correct for t near 1. A more faithful implementation
            # would store/interpolate the FULL char_trajectory over t, not
            # just its final frame -- flagged here as a known simplification
            # to revisit if pretraining doesn't converge well in practice.
            L_char = ((phi_at * u_pred - phi_at * u_char_at) ** 2).mean()

            loss = L_pde + L_char
            loss.backward()
            optimizer.step()

        # --- Stage 2: fine-tune mu_D, mu_R, thresholds on real segmentation ---
        mu_D = torch.tensor(1.0, requires_grad=True, device=device)
        mu_R = torch.tensor(1.0, requires_grad=True, device=device)
        u_t1gd_c = torch.tensor(0.6, requires_grad=True, device=device)
        u_flair_c = torch.tensor(0.35, requires_grad=True, device=device)
        finetune_params = [mu_D, mu_R, u_t1gd_c, u_flair_c] + list(net.parameters())
        optimizer_ft = torch.optim.Adam(finetune_params, lr=lr * 0.1)

        if y_t1gd_real is not None and y_flair_real is not None:
            y_t1gd_full = torch.tensor(np.asarray(y_t1gd_real).astype(np.float32), device=device)
            y_flair_full = torch.tensor(np.asarray(y_flair_real).astype(np.float32), device=device)
        else:
            # Fallback: two thresholds of the SAME density field, not genuinely
            # independent T1Gd/FLAIR signals -- see this function's y_t1gd_real
            # docstring note above.
            y_t1gd_full = torch.tensor((c_followup > 0.5).astype(np.float32), device=device)
            y_flair_full = torch.tensor((c_followup > 0.1).astype(np.float32), device=device)

        for _ in range(n_finetune_iters):
            optimizer_ft.zero_grad()
            xyzt, idx = sample_collocation(n_collocation)
            Pphi_at = gather_at_idx(Pphi_t, idx)
            gradPphi_at = gather_at_idx(grad_Pphi_t, idx)
            phi_at = gather_at_idx(phi_t, idx)
            residual = pde_residual(net, xyzt, x0_nd, mu_D=mu_D, mu_R=mu_R,
                                     script_D=script_D, script_R=script_R,
                                     P_interp=Pphi_at, phi_interp=phi_at,
                                     grad_P_phi_interp=gradPphi_at)
            L_pde = (residual ** 2).mean()

            data_flat_idx = rng.integers(0, np.prod(shape), size=n_data_points)
            data_idx = np.unravel_index(data_flat_idx, shape)
            xyz_data = coords_nd[data_idx]
            t_data = np.ones(n_data_points)
            xyzt_data = torch.tensor(np.concatenate([xyz_data, t_data[:, None]], axis=-1),
                                      dtype=torch.float32, device=device)
            phi_data_at = gather_at_idx(phi_t, data_idx)
            y_t1gd_at = gather_at_idx(y_t1gd_full, data_idx)
            y_flair_at = gather_at_idx(y_flair_full, data_idx)

            L_seg = segmentation_loss(net, xyzt_data, x0_nd, y_t1gd_at, y_flair_at,
                                       u_t1gd_c, u_flair_c, phi_data_at)

            L_constraints = (box_penalty(mu_D, 0.75, 1.25) + box_penalty(mu_R, 0.75, 1.25) +
                              box_penalty(u_t1gd_c, 0.5, 0.8) + box_penalty(u_flair_c, 0.2, 0.5))

            loss = L_pde + 1e-3 * L_seg + L_constraints
            loss.backward()
            optimizer_ft.step()

        # --- final prediction at t=1, evaluated on the real voxel grid ---
        with torch.no_grad():
            flat_idx_all = np.arange(np.prod(shape))
            idx_all = np.unravel_index(flat_idx_all, shape)
            xyz_all = coords_nd[idx_all]
            t_all = np.ones(len(flat_idx_all))
            xyzt_all = torch.tensor(np.concatenate([xyz_all, t_all[:, None]], axis=-1),
                                     dtype=torch.float32, device=device)
            u_pred_flat = hard_ic_output(net, xyzt_all, x0_nd).cpu().numpy()
        u_pred = u_pred_flat.reshape(shape) * phi  # zero outside brain via phase field

        # --- evaluation, matching this project's own dice/coverage convention ---
        pred_binary = u_pred > float(u_t1gd_c.item())
        target_binary = c_followup > 0.5
        intersection = (pred_binary & target_binary).sum()
        dice = (2 * intersection + 1e-6) / (pred_binary.sum() + target_binary.sum() + 1e-6)
        coverage = (intersection + 1e-6) / (target_binary.sum() + 1e-6)
        rmse = float(np.sqrt(np.mean((u_pred - c_followup) ** 2)))

        return {
            "dice": float(dice), "coverage": float(coverage), "rmse": rmse,
            "mu_D": float(mu_D.item()), "mu_R": float(mu_R.item()),
            "Dbar": Dbar, "rhobar": rhobar, "Lbar": Lbar,
        }


# ---------------------------------------------------------------------------
# Smoke test: numpy-only pieces, no torch required (RUN in this dev sandbox)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from generate_synthetic_growth_data import make_synthetic_anatomy

    tissue_mask, D_tensor, affine = make_synthetic_anatomy(shape=(40, 40, 40), seed=0)

    P = compute_tissue_geometry(tissue_mask)
    phi = compute_phase_field(tissue_mask, voxel_size=1.0)
    print(f"P(x) range: [{P.min():.2f}, {P.max():.2f}] (expect 0, 1, or 10 -- gray/bg, gray, white)")
    print(f"phi(x) range: [{phi.min():.3f}, {phi.max():.3f}]")
    print(f"phi mean inside brain: {phi[tissue_mask > 0].mean():.3f} (expect close to 1)")
    print(f"phi mean at tissue_mask==0 (the CARVED-OUT VENTRICLE here, not true background -- "
          f"make_synthetic_anatomy fills the entire array with tissue, no exterior margin, so "
          f"this checks the diffuse-domain field's behavior at an INTERIOR cavity, not a real "
          f"skull boundary): {phi[tissue_mask == 0].mean():.3f} (expect somewhere between 0 and 1, "
          f"since it's small and surrounded by tissue -- NOT the same as a true exterior boundary; "
          f"see the padded test below for that case)")

    # A proper exterior-boundary test: pad the same anatomy with a margin of
    # true background (tissue_mask=0) on all sides, unlike make_synthetic_anatomy's
    # own array which has no such margin.
    pad = 8
    padded_mask = np.zeros(tuple(s + 2 * pad for s in tissue_mask.shape), dtype=int)
    padded_mask[pad:-pad, pad:-pad, pad:-pad] = tissue_mask
    phi_padded = compute_phase_field(padded_mask, voxel_size=1.0)
    print(f"\nPadded test (real exterior margin): phi mean in outer 4 voxels of the padding "
          f"(true background, far from brain): {phi_padded[:4, :, :].mean():.4f} (expect close to 0)")
    print(f"Padded test: phi mean deep inside original brain region: "
          f"{phi_padded[pad + 15:pad + 25, pad + 15:pad + 25, pad + 15:pad + 25].mean():.4f} "
          f"(expect close to 1)")

    r, u = spherical_fisher_kpp_1d(D_nd=1.0, R_nd=1.0)
    # Sanity bound: pure logistic growth (no diffusion loss) from u0=0.1 at
    # rate R_nd=1 for one non-dim time unit gives, via the closed-form
    # logistic solution u(t)=u0*e^(Rt)/(1-u0+u0*e^(Rt)): 0.1*e/(0.9+0.1*e) = 0.232.
    # Diffusion can only REDUCE the peak below this (it spreads mass out, it
    # doesn't concentrate it), so u[0] here should be positive but <= ~0.23,
    # not some much larger value and not ~0 either.
    no_diffusion_bound = 0.1 * np.e / (0.9 + 0.1 * np.e)
    print(f"\n1D spherical solver: u(r=0)={u[0]:.4f} (pure-logistic-growth upper bound at "
          f"R_nd=1, t=1 is {no_diffusion_bound:.4f} -- diffusion should pull this down somewhat, "
          f"not to zero), u(r={r[-1]:.1f})={u[-1]:.4f} (expect ~0, far from tumor)")

    # Two genuinely different test cases -- NOT a rescaling of each other
    # (an earlier version of this test used (15,25) vs (30,50), exactly a 2x
    # rescaling with the identical FLAIR/T1Gd=1.667 ratio in both cases,
    # which cannot actually test size-differentiation: a single
    # (ratio, Lbar) grid point reproduces a FIXED FLAIR/T1Gd shape regardless
    # of Lbar, so two same-shape targets can trivially collapse onto the same
    # best-fit grid point without that being a bug -- caught while debugging
    # this test, not a real finding about the grid search itself).
    Dbar, rhobar, Lbar = estimate_characteristic_params(R_t1gd_seg_mm=10.0, R_flair_seg_mm=15.0)
    print(f"\nCharacteristic params for (T1Gd=10mm, FLAIR=15mm, small/compact tumor): "
          f"Dbar={Dbar:.4f}, rhobar={rhobar:.4f}, Lbar={Lbar:.1f}mm")
    Dbar2, rhobar2, Lbar2 = estimate_characteristic_params(R_t1gd_seg_mm=25.0, R_flair_seg_mm=60.0)
    print(f"Characteristic params for (T1Gd=25mm, FLAIR=60mm, LARGER and more diffuse/invasive "
          f"-- different FLAIR/T1Gd shape, not just a rescaling): "
          f"Dbar={Dbar2:.4f}, rhobar={rhobar2:.4f}, Lbar={Lbar2:.1f}mm")
    print("Sanity check: these two genuinely different-shaped targets should generally resolve "
          "to different (Dbar/rhobar, Lbar) grid points, not the same one.")

    # Geometric centroid of valid tissue, not argwhere's positional-order
    # midpoint (which can land near an array edge -- caught while debugging
    # the FDM test below: an earlier version of this test used
    # valid_voxels[len(valid_voxels)//2], which happened to sit at y=39, the
    # very last index of a 40-voxel axis, depressing phi there via array-edge
    # smoothing effects unrelated to the actual anatomy).
    valid_voxels = np.argwhere(tissue_mask > 0)
    x0_index = tuple(np.round(valid_voxels.mean(axis=0)).astype(int))
    print(f"\nSeed location for FDM test: {x0_index} (tissue={tissue_mask[x0_index]}, "
          f"phi={phi[x0_index]:.3f} -- should be close to 1, a true interior point)")

    trajectory = solve_diffuse_domain_fdm(P, phi, D_gray=0.05, rho=0.05, x0_index=x0_index,
                                           tissue_mask=tissue_mask, voxel_size=1.0,
                                           n_steps=400, dt=0.05, save_every=100)
    print(f"Diffuse-domain FDM trajectory: {len(trajectory)} saved frames, "
          f"initial sum u={trajectory[0].sum():.5f}, final max u={trajectory[-1].max():.5f}, "
          f"final sum u={trajectory[-1].sum():.5f} (expect > initial sum -- net growth)")
    leakage_outside = trajectory[-1][tissue_mask == 0].sum()
    total_mass = trajectory[-1].sum()
    print(f"Mass at tissue_mask==0 (interior ventricle here, see note above -- not a true "
          f"exterior boundary): {leakage_outside:.5f} / {total_mass:.5f} total")


# ---------------------------------------------------------------------------
# TODO checklist:
# 1. [DONE, tested] Tissue geometry P(x), phase field phi(x), 1D spherical
#    solver + characteristic-parameter grid search, diffuse-domain FDM
#    solver -- all numpy-only, run via this file's own smoke test above.
# 2. [IMPLEMENTED, UNTESTED] PINNCoordinateNet, pde_residual,
#    segmentation_loss, run_pinn_baseline_patient() -- no torch in this dev
#    sandbox. Smoke-test on 1-2 real RHUH-GBM patients on the next Kaggle
#    run before trusting this at scale (watch: does mu_D/mu_R end up near 1
#    or pinned at the [0.75,1.25] boundary -- if pinned, widen the
#    characteristic-parameter grid the same way RHO_CANDIDATES was widened
#    for the FEM baseline earlier this project).
# 3. Once (2) is verified, wire run_pinn_baseline_patient() into the
#    notebook's Section 8 CV loop the same way evaluate_model_on_patients()/
#    evaluate_personalized_pi() are, and change Table 3's PINN row from
#    "(cited)" to a live, reproduced number -- update the manuscript text
#    that currently describes PINN as cited-not-reproduced accordingly.
# 4. Wire in the real multi-class (necrosis/edema/enhancing) segmentation
#    once anatomy_and_graph_conversion.py supports it (see
#    TODO_Nature_readiness.md's GliODIL section, same underlying data), so
#    y_t1gd/y_flair become genuinely independent signals instead of two
#    thresholds of the same binary mask.
# 5. Revisit the L_char pretraining simplification noted inline above
#    (matching against only the FINAL frame of char_trajectory, not the
#    full non-dim time history) if pretraining doesn't converge well.
# ---------------------------------------------------------------------------

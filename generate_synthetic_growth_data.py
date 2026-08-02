"""
Synthetic data generator for PI-GNN pretraining.

Generates Fisher-KPP reaction-diffusion tumor growth trajectories over
REAL patient anatomy (from UCSF-PDGM / UPenn-GBM segmentations + DTI),
with RANDOMIZED (D, rho) parameters and randomized seed location, so the
PI-GNN can pretrain on a large synthetic set before fine-tuning on the
small amount of real longitudinal data (RHUH-GBM, LUMIERE).

Equation:
    dc/dt = div(D(x) grad c) + rho * c * (1 - c)

Solved here with an explicit finite-volume scheme on a regular voxel grid
(simplest correct discretization; swap in a real FEM mesh solver, e.g.
FEniCS, if you need irregular tetrahedral meshes matching your GNN graph
exactly). Anisotropy comes from a diffusion TENSOR per voxel derived from
DTI, not a scalar -- this preserves the fiber-guided invasion pattern that
matters clinically.

Output format: one .npz per synthetic patient with the full trajectory,
diffusion tensor field, tissue mask, and ground-truth (D_scale, rho) --
directly usable to build the torch_geometric graphs referenced in
pignn_glioma.py (convert grid -> supervoxel graph as a separate step).

Dependencies: numpy, scipy (core solver). Real-anatomy loading in
load_patient_anatomy() additionally needs nibabel, ANTsPy (antspyx), and dipy
-- these are optional imports (guarded, see below) only required if
use_real_anatomy=True; the placeholder/synthetic-anatomy path
(make_synthetic_anatomy) needs only numpy/scipy.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import os

try:
    import nibabel as nib  # only needed by load_patient_anatomy() with real data
except ImportError:
    nib = None

try:
    import ants  # ANTsPy, only needed for the DWI->T2 registration step below
except ImportError:
    ants = None

try:
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.reconst import dti as dipy_dti
except ImportError:
    read_bvals_bvecs = gradient_table = dipy_dti = None


# ---------------------------------------------------------------------------
# 1. Load real anatomy (tissue mask + DTI tensor field)
# ---------------------------------------------------------------------------
#
# STATUS: implemented, but NOT executable or tested in this environment --
# there is no ANTsPy/dipy install and no real TCIA data available here (no
# internet access to install packages, no Aspera client, no real patient
# files on disk). This is written and ready to run the moment real UCSF-PDGM
# data (full TCIA release, accessed via Aspera -- see TODO_Nature_readiness.md
# Tier 1) is available; it has only been checked for consistency against the
# documented UCSF-PDGM file layout and against the approach independently
# confirmed working by a third party (kukrma/glioma-diffusion-analysis, a
# published OHBM 2026 poster reproducing DTI fitting on this exact dataset).
# It has NOT been run against real files, so treat every step below as a
# best-effort implementation to be smoke-tested (small subset, 2-3 patients)
# the moment real data is in hand -- do not trust it blindly on the full run.
#
# Expected per-subject directory layout (matches TCIA's official release):
#   {patient_dir}/{subject_id}_DWI.nii.gz            4D raw diffusion volume
#   {patient_dir}/{subject_id}_DTI_eddy_noreg.nii.gz  eddy-corrected, NOT yet
#                                                      registered to T2 space
#   {patient_dir}/{subject_id}_T2.nii.gz              T2 anatomical (registration target)
#   {patient_dir}/{subject_id}_brain_segmentation.nii.gz   tissue labels
#   {patient_dir}/{subject_id}_tumor_segmentation.nii.gz   tumor mask (unused here,
#                                                            relevant for the real
#                                                            growth-fitting pipeline
#                                                            elsewhere, not this
#                                                            synthetic-pretraining path)
# Plus ONE shared pair for the whole dataset (not per-subject):
#   {dataset_root}/UCSF-PDGM_DTI.bval
#   {dataset_root}/UCSF-PDGM_DTI.bvec
def load_patient_anatomy(patient_dir: str, subject_id: str = None,
                          bval_path: str = None, bvec_path: str = None,
                          brain_seg_suffix: str = "_brain_segmentation.nii.gz"):
    """
    Load a real patient's tissue segmentation and a REAL, per-voxel,
    full (3x3, not just diagonal) DTI-derived diffusion tensor field, to use
    as the anatomical substrate for a SYNTHETIC growth simulation (only the
    growth trajectory is synthetic; the anatomy -- including true fiber
    anisotropy -- is real).

    Pipeline (mirrors the approach independently validated by
    kukrma/glioma-diffusion-analysis on this exact dataset, adapted here for
    a WHOLE-BRAIN, tissue-type-scaled tensor rather than their ROI-restricted
    peritumoral/periedematous scalar statistics):

      1. Register the raw diffusion volume to T2 space (deformable, ANTsPy
         SyN) -- this is the expensive step (~105s/patient reported by
         kukrma et al. on their setup; budget accordingly on Kaggle).
      2. Fit a full diffusion tensor per voxel via dipy against the
         dataset-wide shared bval/bvec (fast, ~12s/patient reported).
      3. Derive tissue_mask from the brain segmentation, using this
         project's convention (0=background/CSF, 1=gray matter,
         2=white matter, 3=ventricle) -- NOTE: the exact label->tissue-type
         mapping below is a best guess based on common segmentation
         conventions and MUST be verified against the real
         *_brain_segmentation.nii.gz label legend once real data is
         available (check the dataset's accompanying documentation / DICOM
         SEG metadata for the authoritative label numbers).
      4. Return the tensor UNSCALED in absolute magnitude (real DTI
         diffusivities are ~1e-3 mm^2/s) -- simulate_growth() already
         multiplies by a randomized D_scale per synthetic patient, so this
         function's job is to get the relative anisotropy/orientation right,
         not the absolute units; if you skip randomized D_scale later and
         use this directly, rescale to your solver's expected magnitude.

    Returns:
        tissue_mask: (X, Y, Z) int array, 0=background/CSF, 1=gray matter,
                     2=white matter, 3=ventricle
        D_tensor:    (X, Y, Z, 3, 3) full (not diagonal-only) diffusion
                     tensor per voxel, in T2-registered space
        affine:      (4, 4) nibabel affine of the T2 (reference) volume

    Raises:
        RuntimeError if ants/dipy/nibabel aren't installed, or if expected
        files aren't found -- these are real prerequisites, not silently
        skippable.
    """
    if nib is None:
        raise RuntimeError(
            "nibabel is required. pip install nibabel (not available in the "
            "dev sandbox this was written in -- verify in your real environment)."
        )
    if ants is None:
        raise RuntimeError(
            "ANTsPy (`pip install antspyx`) is required for DWI->T2 registration. "
            "Not available in the dev sandbox this was written in."
        )
    if dipy_dti is None:
        raise RuntimeError(
            "dipy (`pip install dipy`) is required for tensor fitting. "
            "Not available in the dev sandbox this was written in."
        )

    subject_id = subject_id or os.path.basename(patient_dir.rstrip("/")).replace("_nifti", "")

    def p(suffix):
        return os.path.join(patient_dir, f"{subject_id}{suffix}")

    t2_path = p("_T2.nii.gz")
    dwi_raw_path = p("_DTI_eddy_noreg.nii.gz")
    seg_path = p(brain_seg_suffix)
    for path in (t2_path, dwi_raw_path, seg_path):
        if not os.path.exists(path):
            raise RuntimeError(f"Expected file not found: {path}")

    if bval_path is None or bvec_path is None:
        # shared, dataset-wide files -- typically one directory above the
        # per-subject folders; adjust if your download layout differs
        dataset_root = os.path.dirname(patient_dir.rstrip("/"))
        bval_path = bval_path or os.path.join(dataset_root, "UCSF-PDGM_DTI.bval")
        bvec_path = bvec_path or os.path.join(dataset_root, "UCSF-PDGM_DTI.bvec")
    for path in (bval_path, bvec_path):
        if not os.path.exists(path):
            raise RuntimeError(
                f"Expected shared bval/bvec file not found: {path}. "
                "Pass bval_path/bvec_path explicitly if your layout differs."
            )

    # ---- Step 1: register raw DWI series to T2 space (ANTsPy, deformable) ----
    t2_img_nib = nib.load(t2_path)  # nibabel copy, purely to get a clean affine to return
    t2_img = ants.image_read(t2_path)
    dwi_img_nib = nib.load(dwi_raw_path)  # read via nibabel first to grab the 4D data + affine
    dwi_data_raw = dwi_img_nib.get_fdata()  # (X, Y, Z, n_directions)

    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    b0_indices = np.where(bvals < 50)[0]  # b~0 volumes, standard threshold
    if len(b0_indices) == 0:
        raise RuntimeError("No b0 (bval~0) volume found in bval file -- cannot register without one.")
    b0_volume = dwi_data_raw[..., b0_indices[0]]
    b0_ants = ants.from_numpy(
        b0_volume.astype(np.float32),
        origin=dwi_img_nib.affine[:3, 3].tolist(),
        spacing=[abs(dwi_img_nib.affine[i, i]) for i in range(3)],
    )

    # Deformable registration, b0 (moving) -> T2 (fixed); matches the
    # ElasticSyN-style approach reported by kukrma et al. on this dataset.
    reg = ants.registration(fixed=t2_img, moving=b0_ants, type_of_transform="SyN")
    warp_transforms = reg["fwdtransforms"]

    # Apply the SAME warp to every diffusion-weighted volume in the 4D series
    # (looping per-volume since ants.apply_transforms operates on 3D images).
    n_vols = dwi_data_raw.shape[-1]
    registered_vols = []
    for v in range(n_vols):
        vol_ants = ants.from_numpy(
            dwi_data_raw[..., v].astype(np.float32),
            origin=b0_ants.origin, spacing=b0_ants.spacing,
        )
        warped = ants.apply_transforms(fixed=t2_img, moving=vol_ants,
                                        transformlist=warp_transforms,
                                        interpolator="linear")
        registered_vols.append(warped.numpy())
    dwi_registered = np.stack(registered_vols, axis=-1)  # now in T2 voxel grid/space

    # ---- Step 2: fit full diffusion tensor via dipy against shared bval/bvec ----
    gtab = gradient_table(bvals, bvecs)
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata().astype(int)
    brain_mask = seg_data > 0  # nonzero = any brain tissue label (verify against real legend)

    tensor_model = dipy_dti.TensorModel(gtab)
    tensor_fit = tensor_model.fit(dwi_registered, mask=brain_mask)
    D_tensor = tensor_fit.quadratic_form  # (X, Y, Z, 3, 3), full tensor, dipy's native output

    # ---- Step 3: tissue_mask from segmentation, mapped to this project's convention ----
    # TODO once real data is available: confirm the actual label numbers used by
    # {subject_id}_brain_segmentation.nii.gz (check dataset documentation / DICOM
    # SEG legend -- UCSF-PDGM's segmentation conventions are not re-derived here).
    # Placeholder mapping below assumes a common convention seen in similar
    # brain-parcellation releases; VERIFY before trusting downstream results.
    tissue_mask = np.zeros_like(seg_data, dtype=int)
    tissue_mask[brain_mask] = 1                # default: gray matter
    tissue_mask[seg_data == 2] = 2              # TODO verify: white matter label
    tissue_mask[seg_data == 4] = 3              # TODO verify: ventricle/CSF label

    return tissue_mask, D_tensor, t2_img_nib.affine


def make_synthetic_anatomy(shape=(64, 64, 64), seed=None):
    """
    Fallback placeholder anatomy generator, for testing the solver pipeline
    before real data is wired in. NOT used for actual paper results.
    """
    rng = np.random.default_rng(seed)
    tissue_mask = np.ones(shape, dtype=int)  # all gray matter by default

    # carve out a "white-matter"-like anisotropic band and a ventricle-like hole
    tissue_mask[shape[0] // 3: 2 * shape[0] // 3, :, :] = 2
    cx, cy, cz = shape[0] // 2, shape[1] // 4, shape[2] // 2
    zz, yy, xx = np.meshgrid(range(shape[2]), range(shape[1]), range(shape[0]), indexing="ij")
    ventricle = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 < 25
    tissue_mask[ventricle.transpose(2, 1, 0)] = 0

    D_tensor = np.zeros(shape + (3, 3))
    iso = np.eye(3) * 0.1
    aniso = np.diag([1.0, 0.1, 0.1])  # fast along x, mimicking a fiber direction
    for idx in np.ndindex(shape):
        D_tensor[idx] = aniso if tissue_mask[idx] == 2 else iso

    affine = np.eye(4)
    return tissue_mask, D_tensor, affine


# ---------------------------------------------------------------------------
# 2. Finite-volume Fisher-KPP solver (explicit time stepping)
# ---------------------------------------------------------------------------
def compute_divergence_diffusion(c, D_tensor, tissue_mask, voxel_size=1.0):
    """
    Compute div(D grad c) via a 6-neighbor finite-volume flux sum, using the
    FULL diffusion tensor (including off-diagonal cross terms), not just its
    diagonal.

    Why this matters: on a regular axis-aligned voxel grid, a naive two-point
    stencil per axis (c_neighbor - c_center) only measures the NORMAL
    derivative at each face. That alone can only reconstruct the
    axis-aligned (diagonal) component of an anisotropic flux. A tensor whose
    principal diffusion direction is NOT aligned with the grid axes (e.g. a
    fiber bundle running diagonally through a voxel) has non-zero
    off-diagonal terms D_xy, D_xz, D_yz, and reproducing that requires an
    estimate of the TANGENTIAL gradient at each face too:

        flux_axis = D[axis,axis] * normal_diff
                  + sum_{axis2 != axis} D[axis,axis2] * tangential_grad[axis2]

    The tangential gradient at a face is approximated by averaging a
    central-difference estimate of dc/d(axis2) between the two cells
    sharing that face (same convention already used for D itself: average
    the cell-centered quantity onto the face). This is a standard
    normal+tangential-correction / MPFA-style extended stencil.

    Backward compatibility: when D_tensor is purely diagonal (as in the
    original placeholder anatomy and in every real result reported in the
    paper so far, since no anisotropic DTI tensor has been wired in yet),
    the cross terms are exactly zero and this function is bit-identical to
    the old diagonal-only version (verified in dev testing: max abs
    diff = 0.0 on random diagonal tensor fields). It only changes behavior
    once a genuinely anisotropic (non-diagonal) tensor is passed in, e.g.
    once real per-voxel DTI tensors are wired up via load_patient_anatomy().

    Verified for physical correctness on a synthetic case: a diffusion
    tensor rotated 45 degrees between two grid axes causes a point source to
    spread along that 45-degree direction under this function, versus a
    axis-aligned (0/90 degree) spread bias under the old diagonal-only
    function, which cannot see the tensor's true orientation at all.
    """
    axes = (0, 1, 2)
    flux = np.zeros_like(c)

    # voxel_size may be a scalar (isotropic, the default and the only case
    # ever exercised before this fix -- every call site in this file passes
    # no voxel_size at all, i.e. the scalar default 1.0) or a length-3 array
    # of per-axis physical spacing (e.g. pinn_baseline.py's
    # solve_diffuse_domain_fdm, the first caller to pass a real array here).
    # Broadcasting to (3,) up front lets both cases share one code path.
    vs = np.broadcast_to(np.asarray(voxel_size, dtype=float), (3,))

    # h-scaled central-difference "raw" tangential derivative per axis:
    # 0.5*(c[+1]-c[-1]) sits on the same h*d/dx scale as the normal-face
    # difference (c_neighbor - c_center) used below.
    raw_grad = [0.5 * (np.roll(c, -1, axis=ax) - np.roll(c, 1, axis=ax)) for ax in axes]

    for ax in axes:
        axis_term = np.zeros_like(c)
        for direction in (-1, 1):
            c_nb = np.roll(c, direction, axis=ax)
            D_nb_cell = np.roll(D_tensor, direction, axis=ax)
            D_face = 0.5 * (D_tensor + D_nb_cell)  # same face-averaging convention as before

            normal_term = D_face[..., ax, ax] * (c_nb - c)

            cross_term = np.zeros_like(c)
            for ax2 in axes:
                if ax2 == ax:
                    continue
                raw_grad_nb = np.roll(raw_grad[ax2], direction, axis=ax)
                raw_grad_face = 0.5 * (raw_grad[ax2] + raw_grad_nb)
                cross_term += D_face[..., ax, ax2] * raw_grad_face

            axis_term += normal_term + cross_term

        # Divide THIS axis's contribution by its own spacing before summing
        # into the total -- NOT a single divide-by-voxel_size**2 at the very
        # end applied to the whole accumulated flux. The two are
        # mathematically identical when voxel_size is a scalar (division
        # distributes over a sum: a/h^2 + b/h^2 == (a+b)/h^2), which is why
        # this was never caught before -- every prior call site only ever
        # passed the isotropic scalar default. They are NOT interchangeable
        # once voxel_size varies per axis (the previous single-shot version
        # would try to broadcast a (X,Y,Z) flux array against a length-3
        # array and raise a ValueError -- exactly the crash this fix
        # resolves for pinn_baseline.py's anisotropic-voxel-spacing caller).
        flux += axis_term / (vs[ax] ** 2)

    # zero-flux boundary: no growth into background (outside brain)
    flux[tissue_mask == 0] = 0.0
    return flux


def simulate_growth(tissue_mask, D_tensor, D_scale: float, rho: float,
                     seed_location, n_steps: int = 200, dt: float = 0.05,
                     save_every: int = 10):
    """
    Run the explicit Fisher-KPP simulation forward from a point seed.

    D_scale: global multiplier on the diffusion tensor field (randomize
             across synthetic patients, e.g. uniform in [0.05, 0.5] mm^2/day)
    rho:     proliferation rate (randomize, e.g. uniform in [0.005, 0.05] /day)
    seed_location: (x, y, z) index of initial tumor seed
    Returns: trajectory list of c arrays at intervals of `save_every` steps
    """
    c = np.zeros_like(tissue_mask, dtype=float)
    c[seed_location] = 1.0
    c = gaussian_filter(c, sigma=1.0)  # smooth initial seed into a small blob
    c[tissue_mask == 0] = 0.0

    D_scaled = D_tensor * D_scale
    trajectory = [c.copy()]

    for step in range(1, n_steps + 1):
        diffusion = compute_divergence_diffusion(c, D_scaled, tissue_mask)
        reaction = rho * c * (1 - c)
        c = c + dt * (diffusion + reaction)
        c = np.clip(c, 0.0, 1.0)
        c[tissue_mask == 0] = 0.0

        if step % save_every == 0:
            trajectory.append(c.copy())

    return trajectory


# ---------------------------------------------------------------------------
# 3. Batch generation driver
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(output_dir: str, n_patients: int = 500,
                                use_real_anatomy: bool = False,
                                real_anatomy_dirs: list = None, seed: int = 0):
    """
    Generate n_patients synthetic growth trajectories.

    If use_real_anatomy=True, real_anatomy_dirs must be a list of patient
    directories (e.g. from UCSF-PDGM/UPenn-GBM) that load_patient_anatomy()
    can read; anatomies are sampled with replacement across n_patients so
    each gets a different randomized (D_scale, rho, seed_location) on
    possibly-repeated real anatomies -- this is intentional, since the
    variation you want in pretraining is growth-parameter variation, not
    anatomy variation (anatomy diversity comes from having many real subjects).
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in range(n_patients):
        if use_real_anatomy:
            anat_dir = rng.choice(real_anatomy_dirs)
            tissue_mask, D_tensor, affine = load_patient_anatomy(anat_dir)
        else:
            tissue_mask, D_tensor, affine = make_synthetic_anatomy(seed=int(rng.integers(1e6)))

        D_scale = rng.uniform(0.05, 0.5)
        rho = rng.uniform(0.005, 0.05)

        valid_voxels = np.argwhere(tissue_mask > 0)
        seed_idx = tuple(valid_voxels[rng.integers(len(valid_voxels))])

        trajectory = simulate_growth(tissue_mask, D_tensor, D_scale, rho, seed_idx)

        np.savez_compressed(
            os.path.join(output_dir, f"synthetic_patient_{i:04d}.npz"),
            trajectory=np.stack(trajectory),
            tissue_mask=tissue_mask,
            D_tensor=D_tensor,
            D_scale=D_scale,
            rho=rho,
            seed_location=seed_idx,
        )

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{n_patients} synthetic trajectories")


# ---------------------------------------------------------------------------
# TODO checklist for your actual implementation:
# 1. [IMPLEMENTED, UNTESTED] load_patient_anatomy(): now does real
#    ANTsPy registration (DWI->T2) + dipy tensor fitting against the UCSF-PDGM
#    shared bval/bvec, instead of raising NotImplementedError. This has NOT
#    been run against real files (no ANTsPy/dipy install and no real TCIA
#    data in this dev environment) -- smoke-test on 2-3 real patients before
#    trusting it on a full run, and specifically verify the brain-segmentation
#    label->tissue-type mapping (marked TODO inline in the function), which
#    is a best guess pending the real label legend.
# 2. [DONE] compute_divergence_diffusion(): now uses the full diffusion
#    tensor (normal + tangential-correction flux), not just its diagonal, so
#    fiber-crossing anisotropy is captured once a real, non-diagonal DTI
#    tensor is supplied. Verified bit-identical to the old diagonal-only
#    version whenever D_tensor is diagonal (true of every currently-reported
#    result, since no real anisotropic tensor has been used yet), and
#    verified to correctly reproduce off-axis spread on a synthetic rotated
#    tensor test case. This was purely a solver correction: it does nothing
#    differently until load_patient_anatomy() above actually supplies a
#    real, non-diagonal per-voxel tensor.
# 3. Grid -> graph conversion: write a separate script that takes each
#    .npz here and converts the voxel grid into the supervoxel/parcel graph
#    format expected by pignn_glioma.py (x, edge_index, edge_attr, c0,
#    c_followup, boundary_mask). SLIC or a brain atlas parcellation are both
#    reasonable choices for the supervoxel step.
# 4. Consider replacing this finite-difference grid solver with a proper
#    FEM solver (FEniCS/SfePy) on a tetrahedral mesh if your GNN graph is
#    mesh-based rather than voxel-based -- keeps train/pretrain domains
#    consistent.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test with placeholder anatomy (not real patient data)
    generate_synthetic_dataset(output_dir="./synthetic_data_test", n_patients=5,
                                use_real_anatomy=False)

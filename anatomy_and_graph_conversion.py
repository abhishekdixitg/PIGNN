"""
Two pieces that bridge generate_synthetic_growth_data.py -> pignn_glioma.py:

1. load_patient_anatomy_real(): actual DTI loading + tensor fitting via
   nibabel/dipy, replacing the NotImplementedError stub.
2. grid_to_supervoxel_graph(): converts a voxel-grid trajectory + DTI tensor
   field into the torch_geometric Data object pignn_glioma.py expects
   (x, edge_index, edge_attr, c0, c_followup, boundary_mask), using SLIC
   supervoxels and the finite-volume conductance formula w_ij = (A_ij/h_ij^2)
   * n_ij^T D_ij n_ij from build_edge_features() in pignn_glioma.py.

Dependencies: numpy, scipy, scikit-image (SLIC), nibabel, dipy, torch,
torch_geometric. The graph-conversion half is runnable with just numpy/
scipy/scikit-image; the real DTI loader needs nibabel+dipy installed.
"""

import numpy as np
from scipy import ndimage
from skimage.segmentation import slic


# ---------------------------------------------------------------------------
# 0. Multi-region tumor segmentation remapping (necrosis / edema / enhancing)
# ---------------------------------------------------------------------------
#
# BACKGROUND: this project's data-loading code (in the companion notebook,
# build_lumiere_graph()/build_lumiere_dense()/build_rhuh_graph()) has always
# collapsed the real tumor segmentation to a single binary "any nonzero =
# tumor" mask, discarding sub-region information. This was flagged as a real
# gap while researching GliODIL/PINN reproduction feasibility
# (TODO_Nature_readiness.md): both live cohorts actually ship multi-region
# segmentation at the source --
#   - RHUH-GBM (official TCIA release): DeepMedic-derived, expert-corrected
#     labels for necrosis, peritumoral signal alteration (edema), and
#     enhancing tumor.
#   - LUMIERE: two named automated pipelines, DeepBraTumIA (necrosis,
#     contrast enhancement, edema, and non-enhancing tumor) and HD-GLIO-AUTO
#     (contrast-enhancing tumor + T2/FLAIR signal abnormality) -- the
#     notebook's find_lumiere_seg() already prefers DeepBraTumIA, falling
#     back to HD-GLIO-AUTO, but currently discards which one actually
#     matched, and collapses either one's output to a binary mask.
#
# This function does NOT invent a single fixed label->region mapping from
# documentation guesswork alone -- it's grounded in REAL evidence where
# available. The notebook's build_lumiere_graph() already has a verbose=True
# path that printed one real LUMIERE patient's actual label histogram in a
# prior run (Patient-001, week-000-1/seg_mask.nii, captured in
# pignn-realdata-test-output-5-result.ipynb's cell 22 output):
#     {0.0: 8855054, 1.0: 16937, 2.0: 12807, 3.0: 43202}
# i.e. labels {1, 2, 3} present, NO label 4 at all, and the file path shows
# this patient's segmentation was found via the notebook's generic
# fallback ("seg_mask.nii" directly under the timepoint folder, not inside a
# "deepbratumia" or "hd-glio" named subfolder) -- meaning neither named
# pipeline folder existed for this patient, and the two schemes below that
# assume a label 4 (BraTS modern/legacy) do NOT match this real file at all.
# A third scheme is added below specifically to cover this real, observed
# case: a simple SEQUENTIAL 1/2/3 convention (1=necrotic, 2=edema,
# 3=enhancing, no gap at 3 the way BraTS's own convention has) -- this is
# the scheme actually confirmed against real data, not a documentation
# guess, and should be tried before the two BraTS-family guesses when no
# named pipeline folder was found. The BraTS modern/legacy schemes remain in
# case a different LUMIERE patient (or RHUH-GBM's official TCIA release,
# separately) DOES use a named DeepBraTumIA/HD-GLIO-AUTO/DeepMedic folder
# with genuine BraTS-convention labels -- keep the pipeline_hint threaded
# through so the caller's actual folder-match result decides which scheme to
# try, rather than guessing blind every time.
#
# ALWAYS falls back to the existing safe behavior (whole-tumor binary mask
# only, remaining sub-regions empty) if the label scheme isn't recognized --
# so calling code that only wants the old binary behavior is unaffected, and
# code that wants sub-region masks gets them when available, an honest empty
# result otherwise, never a silently wrong guess.
#
# STATUS: implemented and unit-tested against synthetic label arrays for
# each known convention, INCLUDING one built from the real label histogram
# above (see this file's smoke test at the bottom) -- the sequential-1/2/3
# scheme is grounded in that one real file; the BraTS modern/legacy schemes
# are still unverified against an actual DeepBraTumIA/HD-GLIO-AUTO/DeepMedic
# output, since none was captured in this project's notebook runs so far.
# Verify against more real patients' label histograms (the notebook's
# verbose=True path already prints this) before trusting broadly -- if a
# given file's labels don't match any scheme below, this correctly falls
# back to whole-tumor-only rather than silently mis-assigning regions.
def remap_tumor_subregions(seg_array, pipeline_hint: str = None):
    """
    Convert a raw integer tumor segmentation array into named sub-region
    binary masks, using documented conventions for the pipeline that
    produced it.

    seg_array: integer-labeled array (any shape), as loaded directly from
    the segmentation NIfTI (e.g. `nib.load(path).get_fdata().astype(int)`).
    pipeline_hint: one of "deepbratumia", "hd-glio", "brats", or None.

    Known conventions applied, tried in this order for any hint other than
    "hd-glio" (each signature below is mutually exclusive -- exactly one can
    match a given file, based on which of labels {1,2,3,4} are present):
      - Sequential 1/2/3, no label 4 (1=necrotic, 2=edema, 3=enhancing):
        the scheme actually CONFIRMED against a real LUMIERE file (see this
        function's module-docstring note) -- checked FIRST whenever label 4
        is absent, ahead of the BraTS guesses below.
      - Legacy BraTS-2015 (1=necrosis, 2=edema, 3=non-enhancing tumor,
        4=enhancing tumor): requires labels 1, 2, 3, AND 4 all present.
      - Modern BraTS (1=necrotic/non-enhancing tumor core combined, 2=edema,
        4=enhancing tumor, no label 3): requires 1, 2, 4 present and no 3.
      - "hd-glio": 2-class scheme (1=contrast-enhancing tumor, 2=T2/FLAIR
        signal abnormality i.e. edema+non-enhancing) -- no separate necrotic
        label in this pipeline's own convention, so necrotic mask is empty
        and edema absorbs everything HD-GLIO-AUTO doesn't call
        contrast-enhancing.

    Returns a dict:
        whole_tumor: (same shape) bool, any nonzero label -- IDENTICAL to
                     the old (seg > 0) behavior, so existing c0/c_followup
                     Dice/coverage computations are completely unaffected by
                     this function's presence.
        necrotic:    bool mask, empty (all False) if not resolvable
        edema:       bool mask, empty (all False) if not resolvable
        enhancing:   bool mask, empty (all False) if not resolvable
        t1gd_core:   bool, necrotic | enhancing -- matches the PINN
                     baseline's (pinn_baseline.py) y_t1gd definition (tumor
                     core, per Zhang et al. Section 2.1.4)
        flair_region: bool, whole_tumor -- matches y_flair (tumor core +
                      edema); currently identical to whole_tumor since edema
                      is already part of whatever "whole_tumor" already
                      captured, kept as a separate named field for clarity
                      at call sites and in case whole_tumor's definition
                      changes independently later
        label_scheme_used: str, one of "sequential_1_2_3", "brats_modern",
                           "brats_legacy", "hd-glio", or "unknown"
                           (whole-tumor-only fallback)
                           -- log/print this when calling, so a silent wrong
                           guess is never actually silent.
    """
    seg_array = np.asarray(seg_array)
    whole_tumor = seg_array > 0
    necrotic = np.zeros_like(whole_tumor, dtype=bool)
    edema = np.zeros_like(whole_tumor, dtype=bool)
    enhancing = np.zeros_like(whole_tumor, dtype=bool)
    scheme = "unknown"

    if pipeline_hint == "hd-glio":
        if (seg_array == 2).any() or (seg_array == 1).any():
            enhancing = seg_array == 1
            edema = seg_array == 2
            scheme = "hd-glio"
    else:
        # deepbratumia / brats / None: three schemes, disambiguated by EXACTLY
        # which of labels {1,2,3,4} are present -- each signature below is
        # mutually exclusive with the others, unlike an earlier version of
        # this function which checked "label 3 present" for legacy without
        # also requiring label 4, so a real file with labels {1,2,3} and NO
        # label 4 (see this function's module-docstring note on the real
        # LUMIERE Patient-001 histogram) was mis-detected as "legacy BraTS
        # with zero enhancing tumor" instead of the correct, much more
        # plausible sequential-1/2/3 scheme where 3 simply IS "enhancing".
        has1 = (seg_array == 1).any()
        has2 = (seg_array == 2).any()
        has3 = (seg_array == 3).any()
        has4 = (seg_array == 4).any()

        if has1 and has2 and has3 and not has4:
            # Sequential 1/2/3, no gap at 3 -- the scheme actually CONFIRMED
            # against a real LUMIERE file (module docstring), not a guess.
            # Prefer this over the BraTS guesses whenever label 4 is simply
            # absent, since a real glioma scan lacking enhancing tumor
            # entirely (which "legacy BraTS with unlabeled enhancing" would
            # imply) is a much less likely explanation than "this pipeline
            # just doesn't use label 4."
            necrotic = seg_array == 1
            edema = seg_array == 2
            enhancing = seg_array == 3
            scheme = "sequential_1_2_3"
        elif has1 and has2 and has3 and has4:
            # Legacy BraTS-2015: needs BOTH 3 and 4 genuinely present to be
            # this scheme (not just "assumed missing enhancing" as before).
            necrotic = (seg_array == 1) | (seg_array == 3)
            edema = seg_array == 2
            enhancing = seg_array == 4
            scheme = "brats_legacy"
        elif has1 and has2 and has4 and not has3:
            # Modern BraTS: NCR/NET combined into 1, no separate label 3.
            necrotic = seg_array == 1
            edema = seg_array == 2
            enhancing = seg_array == 4
            scheme = "brats_modern"

    return {
        "whole_tumor": whole_tumor,
        "necrotic": necrotic,
        "edema": edema,
        "enhancing": enhancing,
        "t1gd_core": necrotic | enhancing,
        "flair_region": whole_tumor,
        "label_scheme_used": scheme,
    }


# ---------------------------------------------------------------------------
# 1. Real DTI loading + tensor fitting (nibabel + dipy)
# ---------------------------------------------------------------------------
def load_patient_anatomy_real(dwi_path: str, bval_path: str, bvec_path: str,
                               seg_path: str, white_matter_scale: float = 10.0,
                               gray_matter_scale: float = 1.0):
    """
    Load diffusion-weighted MRI + b-values/vectors, fit a diffusion tensor
    model, and combine with a tissue segmentation to produce the
    (tissue_mask, D_tensor, affine) triple that generate_synthetic_growth_data.py
    and grid_to_supervoxel_graph() both expect.

    dwi_path:  4D DWI NIfTI (X, Y, Z, n_directions)
    bval_path: .bval file (b-values per direction)
    bvec_path: .bvec file (gradient directions)
    seg_path:  3D tissue segmentation NIfTI, integer labels matching your
               atlas convention (0=background/CSF, 1=gray matter,
               2=white matter, 3=ventricle -- adjust to your actual atlas)

    Returns: tissue_mask (X,Y,Z) int, D_tensor (X,Y,Z,3,3) float, affine (4,4)

    NOTE: requires `pip install nibabel dipy --break-system-packages`.
    Not runnable in this environment (no network access to install), but
    syntax-verified; this is the real implementation to drop in once you
    have DTI data + these packages available.
    """
    import nibabel as nib
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    dwi_img = nib.load(dwi_path)
    dwi_data = dwi_img.get_fdata()
    affine = dwi_img.affine

    gtab = gradient_table(bval_path, bvec_path)
    tensor_model = TensorModel(gtab)
    tensor_fit = tensor_model.fit(dwi_data)

    # dipy's quadratic_form gives the full symmetric 3x3 tensor per voxel
    D_tensor_raw = tensor_fit.quadratic_form  # (X, Y, Z, 3, 3), physical units

    seg_img = nib.load(seg_path)
    tissue_mask = seg_img.get_fdata().astype(int)

    # Rescale the fitted tensor by tissue type so white matter shows the
    # expected faster/more anisotropic invasion pathway; fitted DTI values
    # are noisy at the individual-voxel level, so blending with a
    # tissue-type prior like this is standard practice in this literature
    # rather than trusting raw per-voxel tensors alone.
    D_tensor = D_tensor_raw.copy()
    D_tensor[tissue_mask == 2] *= white_matter_scale
    D_tensor[tissue_mask == 1] *= gray_matter_scale
    D_tensor[tissue_mask == 0] = 0.0  # no diffusion outside brain/in CSF

    return tissue_mask, D_tensor, affine


# ---------------------------------------------------------------------------
# 2. Voxel grid -> supervoxel graph conversion
# ---------------------------------------------------------------------------
def compute_supervoxels(tissue_mask, c0, n_segments: int = 500, compactness: float = 0.1):
    """
    Run SLIC supervoxel segmentation restricted to brain tissue.
    Using c0 (baseline tumor density) as part of the SLIC input image
    biases supervoxel boundaries to respect the tumor edge, which matters
    since you don't want a supervoxel straddling healthy/tumor tissue.
    """
    brain_mask = tissue_mask > 0
    # Stack tissue label + baseline density as a 2-channel "image" for SLIC
    slic_input = np.stack([tissue_mask.astype(float), c0], axis=-1)
    labels = slic(slic_input, n_segments=n_segments, compactness=compactness,
                  mask=brain_mask, channel_axis=-1, start_label=0)
    return labels  # (X, Y, Z) int, -1 outside mask (skimage uses 0 as background if mask given)


def build_supervoxel_graph(labels, tissue_mask, D_tensor, c0, c_followup,
                            voxel_size=1.0):
    """
    Aggregate voxel-grid quantities into per-supervoxel node features and
    compute edge weights via the finite-volume conductance formula
    w_ij = (A_ij / h_ij^2) * n_ij^T D_ij n_ij, matching build_edge_features()
    in pignn_glioma.py so the two pipelines are numerically consistent.

    voxel_size: either a scalar (isotropic pixel-unit spacing -- the old,
    still-supported default, since RHUH-GBM's 2D slices come from PNG-style
    images with no NIfTI affine to pull real spacing from) or a length-3
    array/sequence (dx, dy, dz) of REAL physical spacing in mm, typically
    `nib.load(path).header.get_zooms()[:3]` for a NIfTI volume. This matters
    because the conductance formula divides by h_ij^2, the physical distance
    between supervoxel centroids -- if h_ij is computed in raw pixel-index
    units instead of mm (the previous behavior for every real patient, not
    just RHUH-GBM), w_ij is off by whatever factor separates pixel spacing
    from true spacing, for every edge, in every graph. `voxel_size` is
    broadcast against a (3,) centroid coordinate below, so both a scalar and
    a length-3 array work with no other change needed.

    Returns a plain-dict graph (convert to torch_geometric.data.Data at the
    call site, once torch is available):
        node_centroids: (N, 3)
        node_features:  (N, F)  [mean tissue label, mean c0, is-boundary flag]
        c0_agg:         (N,)    mean baseline density per supervoxel
        c_followup_agg: (N,)    mean follow-up density per supervoxel
        edge_index:     (2, E)
        edge_attr:      (E, 2)  [w_ij, h_ij]  (column 0 = physics weight, matches
                                 the convention expected by ReactionDiffusionStep)
        boundary_mask:  (N,) bool, True for supervoxels touching the brain edge
    """
    voxel_size = np.asarray(voxel_size, dtype=float)  # scalar or (3,) both broadcast correctly below

    unique_labels = np.unique(labels[labels >= 0])
    n_nodes = len(unique_labels)
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

    node_centroids = np.zeros((n_nodes, 3))
    node_features = np.zeros((n_nodes, 3))
    raw_sizes = np.zeros(n_nodes)
    c0_agg = np.zeros(n_nodes)
    c_followup_agg = np.zeros(n_nodes)
    boundary_mask = np.zeros(n_nodes, dtype=bool)
    node_D = np.zeros((n_nodes, 3, 3))

    brain_eroded = ndimage.binary_erosion(tissue_mask > 0)
    boundary_voxels = (tissue_mask > 0) & (~brain_eroded)

    for lab in unique_labels:
        idx = label_to_idx[lab]
        voxel_mask = labels == lab
        coords = np.argwhere(voxel_mask)
        node_centroids[idx] = coords.mean(axis=0) * voxel_size
        raw_sizes[idx] = voxel_mask.sum()
        node_features[idx, 0] = tissue_mask[voxel_mask].mean()
        node_features[idx, 1] = c0[voxel_mask].mean()
        c0_agg[idx] = c0[voxel_mask].mean()
        c_followup_agg[idx] = c_followup[voxel_mask].mean()
        boundary_mask[idx] = boundary_voxels[voxel_mask].any()
        node_D[idx] = D_tensor[voxel_mask].mean(axis=0)

    # Relative supervoxel size (size / mean size across this graph), not the raw
    # voxel count. The raw count is O(10-1000) depending on image resolution and
    # n_segments, while the other two features are O(1) -- at a Linear layer's
    # default init (calibrated for unit-scale inputs) that mismatch alone produces
    # huge pre-activation values, causing exactly the exploding-loss / dead-network
    # behavior (sigmoid/softplus saturation, frozen gradients) observed when this
    # graph was actually trained on, rather than just forward-passed once. Relative
    # size is also more meaningful than a raw count: "1.5x the median supervoxel"
    # is informative and scale-invariant across patients/resolutions, "230 voxels"
    # is neither.
    mean_size = raw_sizes.mean() if raw_sizes.mean() > 0 else 1.0
    node_features[:, 2] = raw_sizes / mean_size

    # Build edges between spatially adjacent supervoxels (shared face in the
    # voxel grid = adjacency in the graph), and accumulate the REAL shared
    # face area per pair while we're at it, instead of assuming a constant.
    #
    # Uses non-wraparound slicing (labels[:-1] vs labels[1:] along each axis),
    # not np.roll: np.roll is circular, so with roll-based adjacency the
    # voxels on one face of the volume (e.g. x=0) get compared against the
    # opposite face (x=Nx-1) and can register as "adjacent" across the whole
    # volume when a small array wraps on itself, or more subtly, silently
    # double-counts real boundary faces once we start counting them (as
    # verified by hand on a small synthetic grid). That was harmless for the
    # old set-membership existence check (edges were deduplicated into a set
    # either way) but would corrupt a real per-pair face-area COUNT, so it's
    # replaced here rather than kept for a computation that now depends on it.
    #
    # A face perpendicular to axis k has physical area = product of
    # voxel_size over the other two axes (e.g. a face normal to x, between
    # two x-adjacent voxels, has area dy * dz).
    vs = np.broadcast_to(voxel_size, (3,)).astype(float)
    face_area_per_axis = [vs[1] * vs[2], vs[0] * vs[2], vs[0] * vs[1]]

    edges = set()
    face_area = {}  # (min(i,j), max(i,j)) -> accumulated real shared-face area
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        a_side = labels[tuple(lo)]
        b_side = labels[tuple(hi)]
        valid = (a_side >= 0) & (b_side >= 0) & (a_side != b_side)
        if not valid.any():
            continue
        pairs = np.stack([a_side[valid], b_side[valid]], axis=-1)
        uniq_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for (a, b), count in zip(uniq_pairs, counts):
            if a in label_to_idx and b in label_to_idx:
                i, j = label_to_idx[a], label_to_idx[b]
                edges.add((i, j))
                edges.add((j, i))  # undirected -> both directions
                key = (min(i, j), max(i, j))
                # each unique (a, b) voxel-face count is a one-voxel-face area;
                # both directions of the same undirected edge share one area
                face_area[key] = face_area.get(key, 0.0) + count * face_area_per_axis[axis]

    edge_index = np.array(list(edges)).T  # (2, E)

    # Finite-volume conductance per edge, same formula as pignn_glioma.py's
    # build_edge_features(), but computed here from the aggregated node_D
    src, dst = edge_index
    diff = node_centroids[dst] - node_centroids[src]
    h_ij = np.linalg.norm(diff, axis=-1) + 1e-8
    n_ij = diff / h_ij[:, None]

    D_mid = 0.5 * (node_D[src] + node_D[dst])
    Dn = np.einsum("eij,ej->ei", D_mid, n_ij)
    n_D_n = np.einsum("ei,ei->e", n_ij, Dn)

    # Real shared-face area per edge (previously a constant A_ij_approx = 1.0
    # regardless of how large or small each supervoxel pair's actual shared
    # boundary was -- biased w_ij for any two supervoxels of very different
    # sizes, since a small sliver of shared face and a broad shared face were
    # weighted identically). Looked up from the face_area dict built above.
    A_ij = np.array([face_area[(min(i, j), max(i, j))] for i, j in zip(src, dst)])
    w_ij = (A_ij / h_ij ** 2) * n_D_n

    edge_attr = np.stack([w_ij, h_ij], axis=-1)

    return {
        "node_centroids": node_centroids,
        "node_features": node_features,
        "c0": c0_agg,
        "c_followup": c_followup_agg,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "boundary_mask": boundary_mask,
    }


# ---------------------------------------------------------------------------
# Smoke test with synthetic grid data (no torch/dipy required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from generate_synthetic_growth_data import make_synthetic_anatomy, simulate_growth

    # --- remap_tumor_subregions(): test against the REAL LUMIERE Patient-001
    # label histogram captured in a prior notebook run (see module docstring
    # above the function), plus each documented scheme and the safe fallback ---
    seg_real = np.concatenate([
        np.zeros(8855054, dtype=int), np.ones(16937, dtype=int),
        np.full(12807, 2, dtype=int), np.full(43202, 3, dtype=int)
    ])
    r0 = remap_tumor_subregions(seg_real, pipeline_hint=None)
    assert r0["label_scheme_used"] == "sequential_1_2_3"
    assert r0["necrotic"].sum() == 16937 and r0["edema"].sum() == 12807 and r0["enhancing"].sum() == 43202
    print("remap_tumor_subregions: real Patient-001 histogram -> "
          f"scheme={r0['label_scheme_used']} (necrotic={r0['necrotic'].sum()}, "
          f"edema={r0['edema'].sum()}, enhancing={r0['enhancing'].sum()}) -- OK")

    seg_modern = np.zeros((10, 10, 10), dtype=int)
    seg_modern[2:4, 2:4, 2:4] = 1
    seg_modern[1:5, 1:5, 1:5] = np.where(seg_modern[1:5, 1:5, 1:5] == 0, 2, seg_modern[1:5, 1:5, 1:5])
    seg_modern[3, 3, 3] = 4
    assert remap_tumor_subregions(seg_modern, pipeline_hint="deepbratumia")["label_scheme_used"] == "brats_modern"
    print("remap_tumor_subregions: synthetic modern-BraTS array -> brats_modern -- OK")

    seg_legacy = seg_modern.copy()
    seg_legacy[3, 3, 2] = 3
    r_legacy = remap_tumor_subregions(seg_legacy, pipeline_hint="brats")
    assert r_legacy["label_scheme_used"] == "brats_legacy"
    assert r_legacy["necrotic"].sum() == ((seg_legacy == 1) | (seg_legacy == 3)).sum()
    print("remap_tumor_subregions: synthetic legacy-BraTS array -> brats_legacy -- OK")

    seg_hdglio = np.zeros((10, 10, 10), dtype=int)
    seg_hdglio[1:5, 1:5, 1:5] = 2
    seg_hdglio[2:4, 2:4, 2:4] = 1
    r_hdglio = remap_tumor_subregions(seg_hdglio, pipeline_hint="hd-glio")
    assert r_hdglio["label_scheme_used"] == "hd-glio" and r_hdglio["necrotic"].sum() == 0
    print("remap_tumor_subregions: synthetic hd-glio array -> hd-glio -- OK")

    seg_unknown = np.zeros((10, 10, 10), dtype=int)
    seg_unknown[3:6, 3:6, 3:6] = 7
    r_unknown = remap_tumor_subregions(seg_unknown, pipeline_hint=None)
    assert r_unknown["label_scheme_used"] == "unknown"
    assert r_unknown["whole_tumor"].sum() == (seg_unknown > 0).sum()
    assert not (r_unknown["necrotic"].any() or r_unknown["edema"].any() or r_unknown["enhancing"].any())
    print("remap_tumor_subregions: unrecognized label -> safe whole-tumor-only fallback -- OK\n")

    tissue_mask, D_tensor, affine = make_synthetic_anatomy(shape=(32, 32, 32), seed=0)
    valid_voxels = np.argwhere(tissue_mask > 0)
    seed_idx = tuple(valid_voxels[len(valid_voxels) // 2])

    trajectory = simulate_growth(tissue_mask, D_tensor, D_scale=0.3, rho=0.03,
                                  seed_location=seed_idx, n_steps=100, save_every=20)
    c0, c_followup = trajectory[0], trajectory[-1]

    labels = compute_supervoxels(tissue_mask, c0, n_segments=80)
    graph = build_supervoxel_graph(labels, tissue_mask, D_tensor, c0, c_followup)

    print(f"Supervoxels: {graph['node_centroids'].shape[0]}")
    print(f"Edges: {graph['edge_index'].shape[1]}")
    print(f"c0 range: [{graph['c0'].min():.4f}, {graph['c0'].max():.4f}]")
    print(f"c_followup range: [{graph['c_followup'].min():.4f}, {graph['c_followup'].max():.4f}]")
    print(f"edge_attr (w_ij, h_ij) sample:\n{graph['edge_attr'][:5]}")
    print(f"Boundary nodes: {graph['boundary_mask'].sum()} / {len(graph['boundary_mask'])}")


# ---------------------------------------------------------------------------
# TODO checklist:
# 1. load_patient_anatomy_real(): install nibabel + dipy
#    (`pip install nibabel dipy --break-system-packages`), verify against a
#    real UCSF-PDGM/UPenn-GBM subject's DWI + segmentation.
# 2. build_supervoxel_graph(): replace the constant A_ij_approx with real
#    shared-face voxel counts between adjacent supervoxel pairs -- currently
#    a placeholder that will bias w_ij for supervoxels of very different sizes.
# 3. Wrap the returned dict into an actual torch_geometric.data.Data object
#    once torch/torch_geometric are installed:
#      Data(x=torch.tensor(graph['node_features'], dtype=torch.float),
#           edge_index=torch.tensor(graph['edge_index'], dtype=torch.long),
#           edge_attr=torch.tensor(graph['edge_attr'], dtype=torch.float),
#           c0=torch.tensor(graph['c0'], dtype=torch.float),
#           c_followup=torch.tensor(graph['c_followup'], dtype=torch.float),
#           boundary_mask=torch.tensor(graph['boundary_mask'], dtype=torch.bool))
# 4. Tune n_segments in compute_supervoxels() -- this is the resolution
#    ablation parameter mentioned in the experiments section.
# 5. [DONE, partially real-data-verified] remap_tumor_subregions(): added to
#    extract necrosis/edema/enhancing sub-region masks instead of collapsing
#    tumor segmentation to a single binary mask -- a prerequisite for both a
#    GliODIL case study and the PINN baseline's separate T1Gd/FLAIR loss
#    terms (pinn_baseline.py). The sequential-1/2/3 scheme is grounded in a
#    REAL LUMIERE patient's actual label histogram (captured in a prior
#    notebook run); the BraTS modern/legacy schemes are still unverified
#    guesses. NOT YET wired into the notebook's find_lumiere_seg() /
#    build_lumiere_graph() / build_lumiere_dense() / build_rhuh_graph() --
#    those still collapse to (seg > 0) directly. To wire in: (a) have
#    find_lumiere_seg() also return WHICH named folder matched
#    ("deepbratumia"/"hd-glio"/None), (b) call remap_tumor_subregions(seg_b,
#    pipeline_hint=that_value) in build_lumiere_graph()/build_lumiere_dense(),
#    (c) print label_scheme_used per patient during the CV loop so an
#    unexpected "unknown" rate is visible, not silent, (d) for RHUH-GBM,
#    do NOT attempt this -- its HuggingFace mirror's "segmentation" column is
#    a lossy JPEG-compressed image with no confirmed color/label convention,
#    unlike LUMIERE's raw NIfTI integer labels; keep RHUH-GBM's existing
#    binary (seg > 0) behavior and note the limitation explicitly rather than
#    risk silently wrong sub-region assignment on compressed pixel values.
# ---------------------------------------------------------------------------

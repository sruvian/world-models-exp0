# Where World Models Keep Their Physics — Code

Reference implementation and analysis. All results run on CPU; no GPU required.

## 1. Setup

Python 3.11+.

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    pip install -e .

CPU torch is pinned in `requirements.txt`. For GPU (optional, not needed):
`pip install -r requirements-gpu.txt`.

Pin BLAS threads before any run — this is what makes CPU execution fast:

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

## 2. Fast reproduction: paper tables from shipped CSVs

Per-checkpoint analysis CSVs ship compressed under `results_csv/`, one folder per
(analysis, architecture) — e.g. `linear_probe_mlp/`, `comparator_rssm/`.

**Before running any aggregator, extract these folders to the repository root.**
The aggregators glob their input folders from the repository root by default; if
the CSV folders are left inside `results_csv/`, the aggregators will find nothing.

    # from the repository root, after extracting results_csv/:
    mv results_csv/* .        # or copy; the aggregators expect <analysis>_<arch>/ at root

Each aggregator parses architecture from the folder name and auto-skips seed
folders. Run from the repository root:

    python aggregators/linear_probe_aggregator.py     # Table 3
    python aggregators/induced_aggregator.py          # Table 4 (encoder)
    python aggregators/comparator_aggregator.py       # Table 4 (operator)
    python aggregators/behavioural_aggregator.py      # Table 5
    python aggregators/jacobian_aggregator.py         # Operator analysis (supp.)
    python aggregators/multilayer_aggregator.py       # Multilayer probing (supp.)
    python aggregators/patch_aggregator.py            # Patching (supp.)
    python aggregators/cross_patch_aggregator.py      # Cross-config patching (supp.)

Multi-seed headline numbers (the reported mean±std):

    python aggregators/seed_aggregator.py

Main figure — combine the `linear_probe_seed_*` and `comparator_seed_*` folders
into one directory and pass it as `--seed_dir` (repeatable):

    python plot_3pan.py --seed_dir <combined_seed_dir> --out panel.pdf

## 3. Full reproduction: from scratch

### 3a. Generate datasets

    python generate_datasets.py --workers 14

Full grid (2 envs × 3 gravities × 3 lengths × 9 seeds × 2 policies) under
`./datasets/`. Fixed settings (see supplement): 500 trajectories/config,
1000 steps/episode, policy seed 35, collection seeds 43–51.

### 3b. Configure training

Base configs are in `trainer_yamls/`. To change common fields, edit the `CHANGES`
dict at the top of `yaml_modifier.py` (uncomment and set the lines you need):

    python yaml_modifier.py --yaml_dir trainer_yamls/

Fields set via `CHANGES`:
  - `model.name`: `WorldModel` (MLP), `WorldModelDMD`, `WorldModelGRU`,
    `WorldModelRSSM`
  - `datasets.regime`: `combined`, `holdg`, `holdl`
  - `checkpointing.save_path`, `checkpointing.logbase_dir`

**DMD-reg:** set `model.name: WorldModelDMD` AND `hyperparams.reg: true`. This
enables the spectral-norm penalty; without it you train plain DMD.

Per-run scalars can instead be passed as flags:

    python yaml_modifier.py --latent 32 --k 15 --seed 3 --beta 0.1

### 3c. Generate the sweep

    python yaml_generator.py --input_dir <base_configs> \
        --output_dir combined_generated \
        --latents 8 16 32 64 --rollouts 1 3 5 15 50 --seeds 0

Constrain `--latents`, `--rollouts`, or `--seeds` for a lighter reproduction.

### 3d. Train

Set the configs path in `run_cseed.sh`, then: `bash run_cseed.sh`

### 3e. Analyze (trained checkpoints → CSVs)

Linear probe (Table 3):

    python -m analysis.probes.linear_probe --models_dir <ckpts> --save_dir <out>
    # RSSM recurrent carry: add --state_type latent_full --probe_target h --roll 1200
    # random-encoder baseline: add --random_init

Multilayer probing (supp.):

    python -m analysis.probes.mult_probe --models_dir <ckpts> --out_dir <out>

Jacobian / operator (supp.):

    python -m analysis.rollout_probes.jacobian_eval --models_dir <ckpts> --out_dir <out>

Patching (supp.) — probe coefficients first:

    python meta_readout.py --models_dir <ckpts> --readout_type probe \
        --save_dir readouts --representation computational
    python meta_patcher.py --readouts_dir readouts --models_dir <ckpts> \
        --save_dir <out> --method activation        # in-config
    python meta_patcher.py --readouts_dir readouts --models_dir <ckpts> \
        --save_dir <out> --method cross_config       # cross-config interchange

Comparator / operator geometry (Table 4, operator half):

    python run_comparator.py --readouts_dir readouts --models_dir <ckpts> --save_dir <out>
    # RSSM rollout variants: add --n_traj 200 --steps 1500

Encoder-induced direction (Table 4, encoder half):

    python run_induced.py --readouts_dir readouts --save_dir <out>

Behavioural law recovery (Table 5; not applicable to RSSM rollout):

    python run_behavioural.py --models_dir <ckpts> --save_dir <out>

Coincident-state (supp.):

    python overlap_test.py --ckpt <checkpoint> \
        --gA 5 --lA 2 --gB 5 --lB 18 --fixed_action 0

Fisher / CRB (Table 2, supp.):

    python fim_identifiability/pooled_crb.py

Then aggregate as in §2.

## 4. Mapping paper results to code

| Paper item | Analysis | Aggregator | Result folders |
|---|---|---|---|
| Table 2 (Fisher/CRB) | `fim_identifiability/pooled_crb.py` | — | — |
| Table 3 (probe recovery) | `analysis.probes.linear_probe` | `linear_probe_aggregator` | `linear_probe_*` |
| Table 4 (encoder) | `run_induced.py` | `induced_aggregator` | `induced_*` |
| Table 4 (operator) | `meta_readout.py` → `run_comparator.py` | `comparator_aggregator` | `comparator_*` |
| Table 5 (behavioural) | `run_behavioural.py` | `behavioural_aggregator` | `behavioural_*` |
| Figure 1 | `plot_3pan.py` | — | `*_seed_*` |
| Jacobian/operator (supp.) | `analysis.rollout_probes.jacobian_eval` | `jacobian_aggregator` | `jacobian_*` |
| Multilayer (supp.) | `analysis.probes.mult_probe` | `multilayer_aggregator` | `multilayer_*` |
| Patching (supp.) | `meta_patcher.py` (both methods) | `patch_aggregator`, `cross_patch_aggregator` | `act_patch_*`, `cross_cfg_*` |
| Coincident-state (supp.) | `overlap_test.py` | — | — |

## 5. Repository layout

    analysis/
      common/          data loading, activations, regime handling
      probes/          linear_probe, mult_probe
      patchers/        activation_patching, cross_cfg, int_gradients
      rollout_probes/  jacobian_eval
      do_checker/      causal checks
    aggregators/       one aggregator per analysis
    fim_identifiability/   Fisher / CRB (pooled_crb.py)
    trainer_yamls/     pre-built training configs
    generate_datasets.py, yaml_modifier.py, yaml_generator.py, run_cseed.sh
    meta_readout.py, meta_patcher.py, run_comparator.py,
    run_behavioural.py, run_induced.py, overlap_test.py, plot_3pan.py
    <analysis>_<arch>/     result CSVs at repo root (see §2)
# Input And Output

`HalluDesign_esmfold2` keeps the HalluDesign optimization loop but supports only
the Biohub ESMFold2 backend.

## Required Inputs

| Argument | Description |
|----------|-------------|
| `--input_file`, `--pdb_list`, or `--random_init_chain_spec` | Input PDB file, a text file containing one PDB path per line, or a no-PDB random-init chain length spec. |
| `--template_path` | HalluDesign template JSON. Protein chains should appear first. |
| `--output_dir` | Output directory. `processing_results.csv` is written here. |

## Common Arguments

| Argument | Description |
|----------|-------------|
| `--mpnn` | `protein_mpnn`, `ligand_mpnn`, or `ligandmpnn_plus_proteinmpnn`. Defaults to `protein_mpnn` for protein-only systems and `ligand_mpnn` when ligand/DNA/RNA inputs are provided. |
| `--num_seqs` | Number of MPNN sequences evaluated after `--design_epoch_begin`. |
| `--num_designs` | Number of independent no-PDB random-init trajectories. Only used with `--random_init_chain_spec`. |
| `--num_recycles` | Number of HalluDesign recycle rounds. |
| `--design_epoch_begin` | Recycle index at which multi-sequence evaluation begins. |
| `--ref_time_steps` | Number of final ESMFold2 denoising steps used to refine from the current coordinates. Default: `6`. |
| `--random_init_chain_spec` | No-PDB random-init protein-chain length spec, for example `A:80`, `A:50-80`, or `B:50-80`. Ligand, DNA, and RNA chains are not randomized in this ESMFold2 runner because the downstream MPNN step redesigns protein sequences. |
| `--seed` | Base random seed. In no-PDB multi-design mode, trajectory seeds are `seed + design_index`. |
| `--fix_chain_index` | Fixed chain IDs, for example `B` or `A B`. |
| `--fix_res_index` | Fixed residues, for example `A12 B35`. |
| `--fix_seq_file` | CSV with `file_path` and optional `fix_res` / `bias` columns. `file_path` can match an input PDB basename such as `monomer.pdb`, or a no-PDB design tag such as `random_init_001`. |
| `--sm` | SMILES string(s) for ligand chains in the template. |
| `--dna` / `--rna` | DNA or RNA sequence(s). |
| `--symmetry_residues` | Residue symmetry groups, for example `A12,B12|A13,B13`. |
| `--symmetry_chains` | Chain symmetry groups, for example `A,B,C`. |
| `--symmetry_segments` | Repeat symmetry for chain A. |
| `--cyclic` | Cyclic residue-index positional encoding for ESMFold2: `1` for first protein chain, `3` for first three protein chains. This does not add a head-tail covalent bond. |

## ESMFold2 Arguments

| Argument | Description |
|----------|-------------|
| `--esmfold2_model_path` | Local ESMFold2 checkpoint path or Hugging Face model name. Defaults to the tested local snapshot under `/storage/caolab/fangmc/cache/huggingface/hub`. |
| `--esmc_model_path` | Local ESMC-6B checkpoint path or Hugging Face model name. Defaults to the tested local snapshot under `/storage/caolab/fangmc/cache/huggingface/hub`. |
| `--esmfold2_num_loops` | Number of ESMFold2 trunk refinement loops. Default: `3`. |
| `--esmfold2_num_sampling_steps` | ESMFold2 diffusion schedule length for full prediction/refinement. Default: `50`, matching the Biohub ESMFold2 GitHub example. Use `0` only to keep the checkpoint config, which is `14` in the tested local snapshot. |
| `--esmfold2_num_diffusion_samples` | Number of ESMFold2 diffusion samples per prediction. Default: `5`; the highest-ranking sample is selected for the next HalluDesign cycle. |
| `--esmfold2_dtype` | Model dtype: `float32`, `bfloat16`, or `float16`. Default: `float32`, matching the tested ESMFold2 setup. |
| `--esmfold2_allow_download` | Allow Hugging Face download. By default the runner uses local files only. |
| `--esmfold2_chunk_size` | Chunk size for memory control. Use `0` to keep the model default. |
| `--esmfold2_device` | `auto`, `cuda`, `cuda:0`, or `cpu`. |

The runner loads ESMFold2 with `load_esmc=False`, then loads ESMC-6B separately
and attaches it to the ESMFold2 model's language-model slot. This matches the
known-working local configuration in
`/storage/caolab/fangmc/code/af3_qa/esmfold2_eval.py`, with an extra internal
`_esmc` assignment needed by the installed Biohub `transformers` forward path.

The default full ESMFold2 setting is `--esmfold2_num_sampling_steps 50`, matching
the Biohub example. `--ref_time_steps` controls how many final denoising steps
are run from the current HalluDesign coordinates. If `--ref_time_steps` is
greater than or equal to `--esmfold2_num_sampling_steps`, the runner ignores the
current coordinates and runs pure ESMFold2 prediction from sequence plus
SMILES/CCD. No-PDB random-init mode also uses pure ESMFold2 prediction for the
first recycle, then feeds the generated PDB into later MPNN/refinement cycles.

## No-PDB Random Init

Use `--random_init_chain_spec` without `--input_file` or `--pdb_list` to start
from random sequences:

```bash
python HalluDesign_esmfold2_run.py \
  --template_path examples/monomer/template_monomer.json \
  --output_dir examples/benchmark/op_random_monomer_50_80 \
  --random_init_chain_spec "A:50-80" \
  --num_designs 20 \
  --mpnn protein_mpnn \
  --num_seqs 8 \
  --num_recycles 20 \
  --design_epoch_begin 1 \
  --ref_time_steps 6
```

For a fixed A-chain scaffold with a random B-chain binder, use
`examples/benchmark/template_protein_binder_random_b.json` and
`--random_init_chain_spec "B:50-80" --fix_chain_index A`.

`--fix_seq_file` also works in no-PDB random-init mode. Example CSV:

```csv
file_path,fix_res,bias
random_init_001,A1 A2 A3,"{'A10': {'W': 1.0, 'Y': 0.5}}"
random_init_002,A5 A6,
```

## Output

Each recycle directory contains MPNN designs, optional ESMFold2 evaluation
folders, and ESMFold2 optimization output:

```text
output_dir/
  processing_results.csv
  recycle_1/
    *_mpnn_eval/
    *_esmfold2_eval/
    *_recycle_1/
      seed_<seed>/
        predictions/
          *_seed_<seed>_sample_0.cif
          *_seed_<seed>_sample_0.pdb
```

The CSV includes MPNN sequence information, ESMFold2 confidence metrics, pLDDT,
pTM/iPTM, and RMSD-style columns where the corresponding structures can be
parsed. No-PDB multi-design runs also include `design_index`, `design_tag`, and
`seed` columns.

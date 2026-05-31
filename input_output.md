# Input And Output

`HalluDesign_esmfold2` keeps the HalluDesign optimization loop but supports only
the Biohub ESMFold2 backend.

## Required Inputs

| Argument | Description |
|----------|-------------|
| `--input_file` or `--pdb_list` | Input PDB file, or a text file containing one PDB path per line. |
| `--template_path` | HalluDesign template JSON. Protein chains should appear first. |
| `--output_dir` | Output directory. `processing_results.csv` is written here. |

## Common Arguments

| Argument | Description |
|----------|-------------|
| `--mpnn` | `protein_mpnn`, `ligand_mpnn`, or `ligandmpnn_plus_proteinmpnn`. Defaults to `protein_mpnn` for protein-only systems and `ligand_mpnn` when ligand/DNA/RNA inputs are provided. |
| `--num_seqs` | Number of MPNN sequences evaluated after `--design_epoch_begin`. |
| `--num_recycles` | Number of HalluDesign recycle rounds. |
| `--design_epoch_begin` | Recycle index at which multi-sequence evaluation begins. |
| `--ref_time_steps` | Number of final ESMFold2 denoising steps used to refine from the current coordinates. Default: `6`. |
| `--fix_chain_index` | Fixed chain IDs, for example `B` or `A B`. |
| `--fix_res_index` | Fixed residues, for example `A12 B35`. |
| `--sm` | SMILES string(s) for ligand chains in the template. |
| `--dna` / `--rna` | DNA or RNA sequence(s). |
| `--symmetry_residues` | Residue symmetry groups, for example `A12,B12|A13,B13`. |
| `--symmetry_chains` | Chain symmetry groups, for example `A,B,C`. |
| `--symmetry_segments` | Repeat symmetry for chain A. |

## ESMFold2 Arguments

| Argument | Description |
|----------|-------------|
| `--esmfold2_model_path` | Local ESMFold2 checkpoint path or Hugging Face model name. Defaults to the tested local snapshot under `/storage/caolab/fangmc/cache/huggingface/hub`. |
| `--esmc_model_path` | Local ESMC-6B checkpoint path or Hugging Face model name. Defaults to the tested local snapshot under `/storage/caolab/fangmc/cache/huggingface/hub`. |
| `--esmfold2_num_loops` | Number of ESMFold2 trunk refinement loops. Default: `3`. |
| `--esmfold2_num_sampling_steps` | ESMFold2 diffusion schedule length for full prediction/refinement. Default: `50`, matching the Biohub ESMFold2 GitHub example. Use `0` only to keep the checkpoint config, which is `14` in the tested local snapshot. |
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
are run from the current HalluDesign coordinates. If a larger value is requested
than the current effective schedule supports, the runner clamps it to the
largest coordinate-refinement value supported by that schedule instead of
switching to pure ESMFold2 prediction. Pure prediction is only used when no
initial coordinates are provided, or for the first cycle of `--random_init`.

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
      seed_123/
        predictions/
          *_seed_123_sample_0.cif
          *_seed_123_sample_0.pdb
```

The CSV includes MPNN sequence information, ESMFold2 confidence metrics, pLDDT,
pTM/iPTM, and RMSD-style columns where the corresponding structures can be
parsed.

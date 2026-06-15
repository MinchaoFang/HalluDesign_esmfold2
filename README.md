# HalluDesign ESMFold2

This is an ESMFold2-only HalluDesign runner. It is separated from the original
HalluDesign repository because Biohub ESMFold2 currently requires Python 3.12,
while the original HalluDesign environment is Python 3.10 based.

The goal is a simple user experience: one Python 3.12 environment and one
command. ESMFold2 is loaded in-process, so there is no per-cycle subprocess
startup or model reload.

A short technical report draft is available in
[`HalluDesign_ESMFold2.pdf`](HalluDesign_ESMFold2.pdf).

## Install

```bash
conda create -n HalluDesign_esmfold2 python=3.12 -y
conda activate HalluDesign_esmfold2
pip install -r requirements.txt
```

`flash-attn` is optional. On older cluster systems such as glibc 2.28, the PyPI
wheel can require a newer glibc and fail to import. In that case, leave it
uninstalled; ESMC can fall back to PyTorch attention. If you want to try a local
source build:

```bash
pip install "torch==2.5.1+cu121" packaging setuptools wheel ninja \
  --extra-index-url https://download.pytorch.org/whl/cu121
pip install "flash-attn==2.7.3"   --no-build-isolation   --no-binary=flash-attn   --no-cache-dir
```

## Example

Monomer optimization:

```bash
python HalluDesign_esmfold2_run.py \
  --input_file examples/monomer/monomer.pdb \
  --template_path examples/monomer/template_monomer.json \
  --output_dir examples/monomer/HalluDesign_op_esmfold2 \
  --num_seqs 2 \
  --num_recycles 10 \
  --ref_time_steps 6
```

Protein-ligand optimization with a SMILES ligand:

```bash
python HalluDesign_esmfold2_run.py \
  --input_file examples/ligand_binder/protein_ligand.pdb \
  --template_path examples/ligand_binder/template_ligand_smiles.json \
  --mpnn ligand_mpnn \
  --sm "C1[C@@H]2[C@H]([C@H]([C@@H](O2)N3C=NC4=C(N=CN=C43)N)O)OP(=O)(O1)O" \
  --output_dir examples/ligand_binder/HalluDesign_op_esmfold2 \
  --num_seqs 2 \
  --num_recycles 10 \
  --ref_time_steps 6
```

No-PDB random-init benchmark:

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
  --ref_time_steps 6 \
  --esmfold2_num_sampling_steps 50 \
  --esmfold2_num_diffusion_samples 5
```

For a fixed A-chain scaffold with a random B-chain binder, use
`examples/benchmark/template_protein_binder_random_b.json` with
`--random_init_chain_spec "B:50-80" --fix_chain_index A`. Additional one-line
benchmark commands are collected in `examples/examples_commands.sh`.

By default the runner uses the local checkpoints that have already been tested
on this machine:

```text
/storage/caolab/fangmc/cache/huggingface/hub/models--biohub--ESMFold2/snapshots/e1e189d0f5fb70c2693da2332eca4443c0ccccd6
/storage/caolab/fangmc/cache/huggingface/hub/models--biohub--ESMC-6B/snapshots/89c554c46a44d825fbfbe3ce2a6bdc539770bdaa
```

For a different machine, pass `--esmfold2_model_path` and `--esmc_model_path`.
The model loader uses `local_files_only=True` by default; add
`--esmfold2_allow_download` only if the environment can access Hugging Face.

## Recommended Use and Limitations

HalluDesign with AF3/Protenix often behaves like a positive feedback loop: MPNN
proposes sequences, the structure model refines the complex, and the confidence
score can gradually improve across cycles. The ESMFold2 backend is different.
ESMFold2/ESMC has learned strong sequence-level protein representations, so even
small amino-acid changes introduced by MPNN can change the sequence prior enough
that a single proposed sequence may not improve together with the current
structure. In practice, ESMFold2-based HalluDesign is less reliable as a
single-sequence optimization loop.

For this reason, the ESMFold2 branch should usually run multiple MPNN sequences
per cycle and evaluate them with ESMFold2 self-consistency before selecting the
next structure. The optimization trajectory can be unstable: confidence scores
may fluctuate across cycles instead of improving steadily, and more cycles may
be needed to reach high-confidence designs. This is especially important for
protein-small molecule complexes, where ligand/interface metrics such as
protein-ligand iPTM may not increase monotonically like AF3 iPTM. Protein-only
optimization is generally more reliable, because ESMFold2 is much stronger at
protein foldability than at small-molecule placement and atom-level interface
optimization.

The main practical value of this branch is speed. ESMFold2 is loaded in-process
and runs quickly, so it is useful for random-init/from-scratch generation,
protein-only optimization, peptide or cyclic-peptide design, fast foldability
screening, and prefiltering candidates before heavier AF3/Protenix validation.
For ligand-binder projects, treat this backend as a fast exploratory or
prefiltering tool rather than a direct replacement for AF3/Protenix interface
optimization.

## Notes

- Use the HalluDesign template JSON files under `examples/*/template_*.json`.
- To start without an input PDB, omit `--input_file` and `--pdb_list`, then set
  `--random_init_chain_spec`, for example `A:80` or `A:50-80`. With
  `--num_designs 20`, the runner starts 20 independent trajectories named
  `random_init_001`, `random_init_002`, and so on. The seed for each trajectory
  is `--seed + design_index`.
- In no-PDB random-init mode, recycle 1 is a pure ESMFold2 prediction from the
  random template sequence. Later recycles use the generated PDB as the current
  structure and run the usual MPNN, ESMFold2 self-consistency, and coordinate
  refinement loop.
- `--esmfold2_num_sampling_steps 0` means use the checkpoint config. For the
  tested local ESMFold2 snapshot this is 14 raw diffusion steps. The runner
  defaults to `--esmfold2_num_sampling_steps 50`, matching the Biohub ESMFold2
  GitHub example.
- `--design_epoch_begin` is 0-based. Before this recycle, the runner still uses
  MPNN, but only with `num_seqs=1`, and skips ESMFold2 self-consistency
  evaluation. From this recycle onward it uses `--num_seqs` and runs multi-seq
  evaluation before choosing the next structure.
- `--cyclic 1` changes ESMFold2 relative residue-index positional encoding for
  the first protein chain to cyclic shortest-path distances. `--cyclic 3` applies
  the same encoding to the first three protein chains. It does not add a
  head-tail covalent bond.
- `--ref_time_steps` means the number of final ESMFold2 denoising steps to run
  from the current HalluDesign structure. If it is greater than or equal to
  `--esmfold2_num_sampling_steps`, the runner ignores the current coordinates
  and runs pure ESMFold2 prediction from sequence plus SMILES/CCD.
- `--esmfold2_num_diffusion_samples` defaults to `5`. HalluDesign decodes all
  ESMFold2 samples, selects the one with the highest ranking score, and writes
  that selected structure to the downstream-compatible `sample_0.cif` path.
- ESMFold2 and ESMC-6B are loaded separately and run in `float32` by default,
  matching the known-working `esmfold2_eval.py` setup.
- `FILE_` ligands, PTMs, and covalent/enzyme-design bonds are not wired in this
  first ESMFold2-only runner.

# Reference
```bibtex
@article {Fang2025.11.08.686881,
	author = {Fang, Minchao and Wang, Chentong and Shi, Jungang and Lian, Fangbai and Jin, Qihan and Wang, Zhe and Zhang, Yanzhe and Cui, Zhanyuan and Wang, YanJun and Ke, Yitao and Han, Qingzheng and Cao, Longxing},
	title = {HalluDesign: Protein Optimization and de novo Design via Iterative Structure Hallucination and Sequence design},
	elocation-id = {2025.11.08.686881},
	year = {2025},
	doi = {10.1101/2025.11.08.686881},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/11/09/2025.11.08.686881},
	eprint = {https://www.biorxiv.org/content/early/2025/11/09/2025.11.08.686881.full.pdf},
	journal = {bioRxiv}
}

@article{Abramson2024,
  author  = {Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and Green, Tim and Pritzel, Alexander and Ronneberger, Olaf and Willmore, Lindsay and Ballard, Andrew J. and Bambrick, Joshua and Bodenstein, Sebastian W. and Evans, David A. and Hung, Chia-Chun and O’Neill, Michael and Reiman, David and Tunyasuvunakool, Kathryn and Wu, Zachary and Žemgulytė, Akvilė and Arvaniti, Eirini and Beattie, Charles and Bertolli, Ottavia and Bridgland, Alex and Cherepanov, Alexey and Congreve, Miles and Cowen-Rivers, Alexander I. and Cowie, Andrew and Figurnov, Michael and Fuchs, Fabian B. and Gladman, Hannah and Jain, Rishub and Khan, Yousuf A. and Low, Caroline M. R. and Perlin, Kuba and Potapenko, Anna and Savy, Pascal and Singh, Sukhdeep and Stecula, Adrian and Thillaisundaram, Ashok and Tong, Catherine and Yakneen, Sergei and Zhong, Ellen D. and Zielinski, Michal and Žídek, Augustin and Bapst, Victor and Kohli, Pushmeet and Jaderberg, Max and Hassabis, Demis and Jumper, John M.},
  journal = {Nature},
  title   = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  year    = {2024},
  volume  = {630},
  number  = {8016},
  pages   = {493–-500},
  doi     = {10.1038/s41586-024-07487-w}
}

@article{bytedance2025protenix,
  title={Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
  author={ByteDance AML AI4Science Team and Chen, Xinshi and Zhang, Yuxuan and Lu, Chan and Ma, Wenzhi and Guan, Jiaqi and Gong, Chengyue and Yang, Jincai and Zhang, Hanyu and Zhang, Ke and Wu, Shenghao and Zhou, Kuangqi and Yang, Yanping and Liu, Zhenyu and Wang, Lan and Shi, Bo and Shi, Shaochen and Xiao, Wenzhi},
  year={2025},
  journal={bioRxiv},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2025.01.08.631967},
  URL={https://www.biorxiv.org/content/early/2025/01/11/2025.01.08.631967},
  elocation-id={2025.01.08.631967},
  eprint={https://www.biorxiv.org/content/early/2025/01/11/2025.01.08.631967.full.pdf},
}

@article{dauparas2023atomic,
  title={Atomic context-conditioned protein sequence design using LigandMPNN},
  author={Dauparas, Justas and Lee, Gyu Rie and Pecoraro, Robert and An, Linna and Anishchenko, Ivan and Glasscock, Cameron and Baker, David},
  journal={Biorxiv},
  pages={2023--12},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}

@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},  
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}

@misc{candido2026language,
  title  = {Language Modeling Materializes a World Model of Protein Biology},
  author = {Candido, Salvatore and Hayes, Thomas and Derry, Alexander and Rao, Roshan
            and Lin, Zeming and Verkuil, Robert and Wu, Bryan and Lee, Jin Sub
            and Bruguera, Elise S. and Keval, Jehan A. and Kopylov, Mykhailo
            and Pak, John E. and Wu, Wesley and Thomas, Neil and Mataraso, Samson
            and Hsu, Alvin and Trotman-Grant, Ashton C. and Fatras, Kilian
            and dos Santos Costa, Allan and Badkundri, Rohil and Ak{\i}n, Halil
            and Oktay, Deniz and Deaton, Jonathan and Montabana, Elizabeth
            and Sitwala, Hrishita and Yu, Yue and Wiggert, Marius
            and Carlin, Dylan Alexander and Goering, Anthony W. and Blazejewski, Tomasz
            and Sandora, McCullen and Hla, Michael and Jia, Tina Z.
            and Kloker, Leon H. and Sofroniew, Nicholas J. and Uehara, Masatoshi
            and Pannu, Jassi and Bachas, Sharrol and Liu, Daniel S.
            and Sercu, Tom and Rives, Alexander},
  year   = {2026},
  url    = {https://biohub.ai/papers/esm_protein.pdf},
  note   = {Preprint}
}

@software{evolutionaryscale_2024,
  author = {{EvolutionaryScale Team}},
  title = {evolutionaryscale/esm},
  year = {2024},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.14219303},
  URL = {https://doi.org/10.5281/zenodo.14219303}
}
```

#!/usr/bin/env bash

# Monomer optimization with Biohub ESMFold2.
python ./HalluDesign_esmfold2_run.py \
  --input_file examples/monomer/monomer.pdb \
  --template_path examples/monomer/template_monomer.json \
  --mpnn protein_mpnn \
  --output_dir "$(pwd)/examples/monomer/HalluDesign_op_esmfold2" \
  --num_seqs 2 \
  --num_recycles 10 \
  --ref_time_steps 6 \
  --esmfold2_num_loops 3 \
  --esmfold2_num_sampling_steps 50 \
  --esmfold2_dtype float32

# Protein-ligand optimization with Biohub ESMFold2.
python ./HalluDesign_esmfold2_run.py \
  --input_file examples/ligand_binder/protein_ligand.pdb \
  --template_path examples/ligand_binder/template_ligand_smiles.json \
  --mpnn ligand_mpnn \
  --sm "C1[C@@H]2[C@H]([C@H]([C@@H](O2)N3C=NC4=C(N=CN=C43)N)O)OP(=O)(O1)O" \
  --output_dir "$(pwd)/examples/ligand_binder/HalluDesign_op_esmfold2" \
  --num_seqs 2 \
  --num_recycles 10 \
  --ref_time_steps 6 \
  --esmfold2_num_loops 3 \
  --esmfold2_num_sampling_steps 50 \
  --esmfold2_dtype float32

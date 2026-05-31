import copy
import json
import os
import re
import shutil
import string
from typing import Dict

from data.op_utility import (
    cdr_process,
    filter_protein_info,
    find_all_sequence_residues_in_pdb,
    random_protein_sequence,
    read_protein_info_from_copied_file,
    residue_to_str,
)
from data.utility import (
    convert_cif_to_pdb,
    find_pocket_residues_based_on_distance,
    generate_A_chain_modulo_symmetry,
    generate_cross_chain_symmetry,
    get_chain_sequence,
)
from eval.evaluation import (
    process_confidence_metrics_esmfold2,
    run_mpnn_evaluation,
    self_consistency_esmfold2,
)


def esmfold2_op_eval(
    pdb_file: str,
    cycle: int,
    output_dir: str,
    template_path: str,
    template_for_eval,
    mpnn_model,
    mpnn_config_dict,
    designer_model,
    ref_time_steps,
    chain_types,
    fixed_chains,
    fixed_residues,
    bais_per_residues,
    metrics,
    symmetry_residues,
    symmetry_chains,
    symmetry_segments,
    sm,
    ccd,
    dna,
    rna,
    cdr,
    framework_seq,
    design_begin,
    chain_number_list_cdr,
    cyclic,
    random_init,
    run_esmfold2: bool = True,
) -> Dict:
    """Run one HalluDesign optimization cycle with ESMFold2."""
    copied_file = pdb_file
    try:
        backend_name = getattr(designer_model, "backend_name", "ESMFold2")
        target_dir = os.path.join(output_dir, f"recycle_{cycle + 1}")
        os.makedirs(target_dir, exist_ok=True)

        source_file = copy.deepcopy(pdb_file)
        metrics["cycle"] = cycle
        pdb_tag = os.path.basename(pdb_file).lower().replace(".pdb", "")
        pdb_tag = re.sub(r"_recycle_.*", "", pdb_tag)
        metrics["file_name"] = pdb_tag
        copied_file = os.path.join(target_dir, f"{metrics['file_name']}_recycle_{cycle + 1}.pdb")

        chain_number_list = []
        if cdr and cycle == 0:
            chain_number_list = cdr_process(cdr, source_file, copied_file)
        elif cdr and cycle > 0:
            chain_number_list = chain_number_list_cdr
            shutil.copy(source_file, copied_file)
        else:
            shutil.copy(source_file, copied_file)
        chain_number_list_cdr = copy.deepcopy(chain_number_list)

        metrics["origin_path"] = os.path.join(
            output_dir,
            "recycle_1",
            f"{metrics['file_name']}_recycle_1.pdb",
        )

        protein_info = read_protein_info_from_copied_file(copied_file)
        if framework_seq:
            print(f"framework_seq {framework_seq}")
            fixed_residues = find_all_sequence_residues_in_pdb(copied_file, framework_seq)

        filtered_info = filter_protein_info(
            protein_info,
            fixed_chains,
            fixed_residues,
            chain_number_list,
        )
        fixed_residues_for_mpnn = [
            f"{chain}{residue_to_str(res.id)}"
            for chain, residues in filtered_info.items()
            for res in residues
        ]

        if symmetry_chains:
            symmetry_residues = generate_cross_chain_symmetry(protein_info, symmetry_chains)
        if symmetry_segments:
            symmetry_residues = generate_A_chain_modulo_symmetry(
                protein_info,
                int(symmetry_segments),
            )

        pocket_res = []
        if sm or ccd or dna or rna:
            pocket_res = find_pocket_residues_based_on_distance(
                pdbfile=copied_file,
                cutoff=8.0,
            )
            print(f"pocket residues: {len(pocket_res)}")

        print("fixed residues for MPNN", fixed_residues_for_mpnn)
        metrics["fixed_residues_for_MPNN_len"] = len(fixed_residues_for_mpnn)
        mpnn_dir = copied_file.replace(".pdb", "_mpnn_eval")
        metrics = run_mpnn_evaluation(
            copied_file,
            mpnn_model,
            mpnn_config_dict,
            fixed_residues_for_mpnn,
            pocket_res,
            mpnn_dir,
            symmetry_residues,
            metrics,
            cyclic,
            cycle,
            bais_per_residues,
        )

        print(f"design begin {design_begin}")
        if design_begin:
            print(f"{backend_name} evaluation")
            eval_dir = copied_file.replace(".pdb", f"{backend_name.lower()}_eval")
            os.makedirs(eval_dir, exist_ok=True)
            template_path_for_eval = template_for_eval or template_path
            metrics = self_consistency_esmfold2(
                scaffold_path=copied_file,
                designer_model=designer_model,
                output_dir=eval_dir,
                template_path=template_path_for_eval,
                sm=sm,
                ccd=ccd,
                dna=dna,
                rna=rna,
                chain_types=chain_types,
                pocket_res=pocket_res,
                fixed_chains=fixed_chains,
                fixed_residues_for_MPNN=fixed_residues_for_mpnn,
                cyclic=cyclic,
                metrics=metrics,
                random_init=random_init,
            )
        else:
            print(f"no {backend_name} evaluation")

        if not run_esmfold2:
            return metrics, copied_file, chain_number_list_cdr

        with open(template_path, "r") as handle:
            input_json = json.load(handle)
        tag = f"{pdb_tag}_recycle_{cycle + 1}"
        json_path = os.path.join(target_dir, f"{metrics[0]['file_name']}.json")
        copied_file = metrics[0]["packed_path"]

        print(f"begin {backend_name} optimization")
        input_json[0]["name"] = tag.replace(".pdb", "")

        count = 0
        sm_count = 0
        rna_count = 0
        dna_count = 0
        protein_count = 0
        chain_labels = string.ascii_uppercase[:10]

        for chain in chain_types:
            if chain == "protein":
                chain_id = chain_labels[protein_count]
                if chain_id not in fixed_chains:
                    if random_init and cycle == 0:
                        sequence = random_protein_sequence(
                            get_chain_sequence(copied_file, chain_id),
                            fixed_residues_for_mpnn,
                            chain_id,
                        )
                    else:
                        sequence = get_chain_sequence(copied_file, chain_id)
                    input_json[0]["sequences"][count]["proteinChain"]["sequence"] = sequence

                    if int(symmetry_segments) >= 2:
                        segment_count = int(symmetry_segments)
                        seq_segment = len(sequence) // segment_count
                        input_json[0]["sequences"][count]["proteinChain"]["sequence"] = (
                            sequence[:seq_segment] * segment_count
                        )
                protein_count += 1

            elif chain == "ligand":
                ligand_block = input_json[0]["sequences"][count]["ligand"]
                if sm_count < len(ccd):
                    ligand_block.pop("ligand", None)
                    ligand_block.pop("smiles", None)
                    ligand_block["ccdCodes"] = [ccd[sm_count]]
                elif sm_count < len(sm):
                    ligand_block.pop("ccd", None)
                    ligand_block.pop("ccdCodes", None)
                    ligand_block["ligand"] = sm[sm_count]
                sm_count += 1

            elif chain == "dna":
                if dna_count < len(dna):
                    input_json[0]["sequences"][count]["dnaSequence"]["sequence"] = dna[dna_count]
                dna_count += 1

            elif chain == "rna":
                if rna_count < len(rna):
                    input_json[0]["sequences"][count]["rnaSequence"]["sequence"] = rna[rna_count]
                rna_count += 1

            count += 1

        count_tuple = [count, protein_count, sm_count, rna_count, dna_count]

        with open(json_path, "w") as handle:
            json.dump(input_json, handle, indent=2)

        if cycle == 0 and random_init:
            print(f"{backend_name} full prediction from random-initialized sequence")
            results_op = designer_model.predict(
                input_json_path=json_path,
                dump_dir=target_dir,
                seed=123,
            )
        elif (
            getattr(designer_model, "num_sampling_steps", None) is not None
            and int(ref_time_steps) >= int(designer_model.num_sampling_steps)
        ):
            print(
                f"{backend_name} full prediction from sequence/SMILES because "
                f"ref_time_steps={int(ref_time_steps)} >= "
                f"esmfold2_num_sampling_steps={int(designer_model.num_sampling_steps)}"
            )
            results_op = designer_model.predict(
                input_json_path=json_path,
                dump_dir=target_dir,
                seed=123,
            )
        else:
            print(
                f"{backend_name} coordinate refinement: requested "
                f"{int(ref_time_steps)} denoising steps"
            )
            print(json_path, copied_file)
            results_op = designer_model.predict(
                input_json_path=json_path,
                dump_dir=target_dir,
                seed=123,
                input_atom_array_path=copied_file,
                diffusion_steps=ref_time_steps,
            )

        if results_op:
            cif_path = os.path.join(
                target_dir,
                tag,
                "seed_123",
                "predictions",
                f"{tag}_seed_123_sample_0.cif",
            )
            metrics = process_confidence_metrics_esmfold2(
                results_op,
                cif_path,
                copied_file,
                metrics[0]["origin_path"],
                metrics,
                pocket_res,
                chain_types,
                fixed_chains,
                count_tuple,
            )
            pdb_output = cif_path.replace(".cif", ".pdb")
            if convert_cif_to_pdb(cif_path, pdb_output):
                return metrics, pdb_output, chain_number_list_cdr
            print("CIF to PDB conversion failed; continuing with the original file")
            return metrics, copied_file, chain_number_list_cdr

        print(f"{backend_name} processing failed; continuing with the original file")
        if isinstance(metrics, list):
            for metric in metrics:
                metric["HalluDesign_Status"] = f"{backend_name}_failed"
        else:
            metrics["HalluDesign_Status"] = f"{backend_name}_failed"
        return metrics, copied_file, chain_number_list_cdr

    except Exception as exc:
        print(f"Failed to process the file: {exc}")
        return metrics, copied_file, chain_number_list_cdr

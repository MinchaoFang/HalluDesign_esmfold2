import copy
import os
import string
from typing import Any, Dict

import numpy as np

from data.utility import (
    calculate_average_b_factor,
    calculate_bfactor_averages_from_list,
    calculate_ca_rmsd,
    calculate_plddt_avg,
    calculate_weights,
    parse_symmetry_residues,
)


def self_consistency_esmfold2(
    scaffold_path,
    designer_model,
    output_dir,
    template_path,
    sm,
    ccd,
    dna,
    rna,
    chain_types,
    pocket_res,
    fixed_chains,
    fixed_residues_for_MPNN,
    cyclic,
    metrics,
    random_init=False,
):
    metrics_to_tile = []
    seq_count = 0
    print(f"ESMFold2 self-consistency evaluation for {len(metrics)} MPNN sequence(s)")
    for metric in metrics:
        seq = metric["mpnn_sequence"]
        print(
            f"  Evaluating MPNN sequence {seq_count + 1}/{len(metrics)} with ESMFold2; "
            f"packed_path={metric.get('packed_path')}"
        )

        with open(template_path, "r") as handle:
            input_json = copy.deepcopy(__import__("json").load(handle))

        scaffold_basename = os.path.splitext(os.path.basename(scaffold_path))[0]
        input_json[0]["name"] = f"{scaffold_basename}_{seq_count}"
        json_path = os.path.join(output_dir, f"{scaffold_basename}_{seq_count}.json")

        count = 0
        sm_count = 0
        rna_count = 0
        dna_count = 0
        protein_count = 0
        chain_labels = string.ascii_uppercase[:10]
        seq_by_chain = seq.split(":")

        for chain in chain_types:
            if chain == "protein":
                if chain_labels[protein_count] not in fixed_chains:
                    input_json[0]["sequences"][count]["proteinChain"]["sequence"] = seq_by_chain[protein_count]
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

        import json

        with open(json_path, "w") as handle:
            json.dump(input_json, handle, indent=2)

        print(f"    ESMFold2 eval input: {json_path}")
        results_eval = designer_model.predict(
            input_json_path=json_path,
            dump_dir=output_dir,
            seed=123,
        )

        max_ranking = _best_summary_index(results_eval)
        summary = results_eval["summary_confidence"][max_ranking]
        chain_iptm = summary["chain_iptm"]
        chain_ptm = summary["chain_ptm"]
        chain_pair_iptm = summary["chain_pair_iptm"]

        tag = f"{scaffold_basename}_{seq_count}".lower()
        cif_path = os.path.join(output_dir, tag, "seed_123", "predictions", f"{tag}_seed_123_sample_0.cif")
        chain_labels = string.ascii_uppercase[:count]

        metric["eval_status"] = "success"
        metric["eval_path"] = cif_path
        metric["prediction_model"] = results_eval.get("model_name", "ESMFold2")
        metric["eval_plddt"] = calculate_average_b_factor(cif_path, [f"{label}" for label in chain_labels])
        metric["eval_iptm"] = _summary_scalar(summary, "iptm", chain_iptm)
        metric["eval_ptm"] = _summary_scalar(summary, "ptm", chain_ptm)
        metric["eval_pae"] = None
        metric["eval_pde"] = None
        metric["eval_ipae"] = None
        metric["eval_ipde"] = None

        if fixed_residues_for_MPNN:
            metric["eval_plddt_fix"], metric["eval_plddt_redes"] = calculate_bfactor_averages_from_list(
                cif_path,
                fixed_residues_for_MPNN,
            )

        for i, label in enumerate(chain_labels):
            metric[f"eval_{label}_plddt"] = calculate_average_b_factor(cif_path, [label])
            metric[f"eval_{label}_ptm"] = _indexed_value(chain_ptm, i)

        try:
            rmsd_result = calculate_ca_rmsd(cif_path, scaffold_path, fixed_chains)
            _add_eval_rmsd_metrics(metric, rmsd_result, chain_types, chain_labels, fixed_chains)
        except Exception:
            print("rmsd wrong")

        if sm_count != 0 or rna_count != 0 or dna_count != 0:
            metric["eval_key_res_plddt"] = calculate_plddt_avg(cif_path, pocket_res)
            _add_interface_eval_metrics(
                metric,
                chain_pair_iptm,
                protein_count,
                count,
                chain_labels,
            )

        print(
            f"    ESMFold2 eval result {seq_count + 1}: "
            f"eval_plddt={metric.get('eval_plddt')}, "
            f"eval_iptm={metric.get('eval_iptm')}, "
            f"eval_ptm={metric.get('eval_ptm')}, "
            f"ranking_score={_summary_scalar(summary, 'ranking_score', None)}, "
            f"eval_path={metric.get('eval_path')}"
        )
        seq_count += 1
        metrics_to_tile.append(metric)

    metrics_to_tile.sort(key=lambda item: item["eval_plddt"], reverse=True)
    if metrics_to_tile:
        best_metric = metrics_to_tile[0]
        print(
            "ESMFold2 self-consistency selected: "
            f"packed_path={best_metric.get('packed_path')}, "
            f"eval_plddt={best_metric.get('eval_plddt')}, "
            f"eval_iptm={best_metric.get('eval_iptm')}, "
            f"eval_ptm={best_metric.get('eval_ptm')}"
        )
    return metrics_to_tile


def run_mpnn_evaluation(
    scaffold_path,
    mpnn_model,
    mpnn_config_dict,
    fixed_residues_for_MPNN,
    pocket_res,
    output_dir,
    symmetry_residues,
    metrics,
    cyclic,
    cycle,
    bais_per_residues=None,
):
    pocket_res_to_fix = " ".join(f"{residue}" for residue in fixed_residues_for_MPNN)
    non_pocket_to_fix = " ".join(f"{residue}" for residue in set(fixed_residues_for_MPNN + pocket_res))

    weights_str = ""
    if symmetry_residues:
        residue_groups = parse_symmetry_residues(symmetry_residues)
        weights = calculate_weights(residue_groups)
        weights_str = "|".join(
            ",".join([str(weight)] * len(group))
            for weight, group in zip(weights, residue_groups)
        )

    bias_AA = ""
    if cyclic:
        bias_AA = "D:0.5,E:0.5,H:0.5,K:0.5,R:0.5,W:-0.5,L:-0.5,I:-0.5,F:-0.5,M:-0.5,V:-0.5,Y:-0.5"

    if mpnn_config_dict["model_name"] == "ligandmpnn_plus_proteinmpnn":
        sequences, packed_paths = run_Ligandmpnn_plus_proteinmpnn_evaluation(
            mpnn_model,
            scaffold_path,
            mpnn_config_dict,
            pocket_res_to_fix,
            non_pocket_to_fix,
            weights_str,
            output_dir,
            symmetry_residues,
        )
    else:
        sequences, packed_paths = run_purempnn_evaluation(
            mpnn_model,
            scaffold_path,
            mpnn_config_dict,
            pocket_res_to_fix,
            weights_str,
            output_dir,
            bias_AA,
            symmetry_residues,
            bais_per_residues,
        )

    metrics_to_tile = []
    metrics_to_copy = copy.deepcopy(metrics)
    for seq, packed in zip(sequences, packed_paths):
        metric = copy.deepcopy(metrics_to_copy)
        metric["mpnn_model"] = mpnn_config_dict["model_name"]
        metric["packed_path"] = packed
        metric["mpnn_sequence"] = seq
        metric["eval_status"] = "Not run"
        metrics_to_tile.append(metric)
    return metrics_to_tile


def run_Ligandmpnn_plus_proteinmpnn_evaluation(
    mpnn_model,
    scaffold_path,
    mpnn_config_dict,
    pocket_res_to_fix,
    non_pocket_to_fix,
    weights_str,
    output_dir,
    symmetry_residues,
):
    ligand_mpnn, protein_mpnn = mpnn_model
    output_dir_mpnn = os.path.join(output_dir, mpnn_config_dict["model_name"])
    sequences_stack, pdb_path_stack = ligand_mpnn.single_protein_mpnn_design(
        scaffold_path=scaffold_path,
        output_dir_mpnn=output_dir_mpnn,
        numbers_seqs=mpnn_config_dict["num_seqs"],
        chains_to_design="",
        fixed_res=pocket_res_to_fix,
        redesigned_residues="",
        symmetry_residues=symmetry_residues,
        weights_str=weights_str,
    )

    sequences = []
    packed_paths = []
    for pdb_path in pdb_path_stack:
        protein_sequences, protein_paths = protein_mpnn.single_protein_mpnn_design(
            scaffold_path=pdb_path,
            output_dir_mpnn=output_dir_mpnn,
            numbers_seqs=1,
            chains_to_design="",
            fixed_res=non_pocket_to_fix,
            redesigned_residues="",
            symmetry_residues=symmetry_residues,
            weights_str=weights_str,
        )
        sequences.append(protein_sequences[0])
        packed_paths.append(protein_paths[0])

    return sequences, packed_paths


def run_purempnn_evaluation(
    mpnn_model,
    scaffold_path,
    mpnn_config_dict,
    pocket_res_to_fix,
    weights_str,
    output_dir,
    bias_AA,
    symmetry_residues,
    bais_per_residues,
):
    output_dir_mpnn = os.path.join(output_dir, mpnn_config_dict["model_name"])
    return mpnn_model.single_protein_mpnn_design(
        scaffold_path=scaffold_path,
        output_dir_mpnn=output_dir_mpnn,
        numbers_seqs=mpnn_config_dict["num_seqs"],
        chains_to_design="",
        fixed_res=pocket_res_to_fix,
        redesigned_residues="",
        symmetry_residues=symmetry_residues,
        bais_per_residues=bais_per_residues,
        input_bias_AA=bias_AA,
        weights_str=weights_str,
    )


def process_confidence_metrics_esmfold2(
    results_op,
    cif_path: str,
    copied_file: str,
    scaffold_path,
    metrics_tile,
    pocket_res,
    chain_types,
    fixed_chains,
    count_tuple,
) -> Dict:
    try:
        max_ranking = _best_summary_index(results_op)
        summary = results_op["summary_confidence"][max_ranking]
        data = results_op["full_data"][max_ranking]

        chain_iptm = summary["chain_iptm"]
        chain_ptm = summary["chain_ptm"]
        chain_pair_iptm = summary["chain_pair_iptm"]
        pae = np.array(data["token_pair_pae"], dtype=np.float32)
        chain_ids = np.array(data["token_asym_id"])

        interface_mask = chain_ids[:, None] != chain_ids[None, :]
        global_ipae = float(pae[interface_mask].mean()) if interface_mask.any() else None

        count, protein_count, sm_count, rna_count, dna_count = count_tuple
        chain_labels = string.ascii_uppercase[:count]

        metrics = {
            "op_cif_path": cif_path,
            "HalluDesign_Status": "ESMFold2_success",
            "op_iptm": _summary_scalar(summary, "iptm", chain_iptm),
            "op_ptm": _summary_scalar(summary, "ptm", chain_ptm),
            "op_pae": float(pae.mean()) if pae.size else None,
            "op_pde": None,
            "op_ipae": global_ipae,
            "op_ipde": None,
            "op_plddt": calculate_average_b_factor(cif_path, [f"{label}" for label in chain_labels]),
        }

        for i, label in enumerate(chain_labels):
            metrics[f"op_{label}_plddt"] = calculate_average_b_factor(cif_path, [label])
            metrics[f"op_{label}_ptm"] = _indexed_value(chain_ptm, i)

        try:
            rmsd_result = calculate_ca_rmsd(cif_path, copied_file, fixed_chains)
            rmsd_result_to_origin = calculate_ca_rmsd(cif_path, scaffold_path, fixed_chains)
            _add_op_rmsd_metrics(
                metrics,
                rmsd_result,
                rmsd_result_to_origin,
                chain_types,
                chain_labels,
                fixed_chains,
            )
        except Exception:
            print("rmsd wrong")

        if sm_count != 0 or rna_count != 0 or dna_count != 0:
            metrics["op_key_res_plddt"] = calculate_plddt_avg(cif_path, pocket_res)
            _add_interface_op_metrics(
                metrics,
                chain_pair_iptm,
                pae,
                chain_ids,
                protein_count,
                count,
                chain_labels,
            )

        for metric in metrics_tile:
            metric.update(metrics)
        return metrics_tile

    except Exception as exc:
        print(f"Failed to process ESMFold2 metrics: {exc}")
        return metrics_tile


def _best_summary_index(results: dict[str, Any]) -> int:
    max_ranking = 0
    max_score = -float("inf")
    for idx, summary in enumerate(results["summary_confidence"]):
        ranking_score = float(summary["ranking_score"])
        if ranking_score > max_score:
            max_score = ranking_score
            max_ranking = idx
    return max_ranking


def _to_float(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _mean_value(value):
    if value is None:
        return None
    if hasattr(value, "mean"):
        return _to_float(value.mean())
    array = np.asarray(value, dtype=np.float32)
    return float(array.mean()) if array.size else None


def _summary_scalar(summary, key, fallback):
    value = summary.get(key)
    if value is not None:
        return _to_float(value)
    return _mean_value(fallback)


def _indexed_value(value, index):
    if value is None:
        return None
    try:
        return _to_float(value[index])
    except Exception:
        return None


def _matrix_value(matrix, row, col):
    try:
        return _to_float(matrix[row][col])
    except Exception:
        return None


def _add_eval_rmsd_metrics(metric, rmsd_result, chain_types, chain_labels, fixed_chains):
    protein_idx = 0
    ligand_idx = 0
    for idx, chain in enumerate(chain_types):
        label = chain_labels[idx]
        if chain == "protein":
            source_idx = idx if fixed_chains else protein_idx
            metric[f"eval_protein_{label}_rmsd"] = rmsd_result["protein_rmsd"][source_idx]
            protein_idx += 1
        elif chain == "ligand":
            metric[f"eval_ligand_{label}_rmsd"] = rmsd_result["ligand_rmsd"][ligand_idx]
            metric[f"eval_atom_distances_{label}"] = rmsd_result["atom_distances"][ligand_idx]
            ligand_idx += 1
        elif chain == "dna":
            metric[f"eval_dna_{label}_rmsd"] = rmsd_result["dna_rmsd"]
        elif chain == "rna":
            metric[f"eval_rna_{label}_rmsd"] = rmsd_result["rna_rmsd"]


def _add_op_rmsd_metrics(metrics, rmsd_result, rmsd_origin, chain_types, chain_labels, fixed_chains):
    protein_idx = 0
    ligand_idx = 0
    for idx, chain in enumerate(chain_types):
        label = chain_labels[idx]
        if chain == "protein":
            source_idx = idx if fixed_chains else protein_idx
            metrics[f"op_protein_{label}_rmsd"] = rmsd_result["protein_rmsd"][source_idx]
            metrics[f"origin_protein_{label}_rmsd"] = rmsd_origin["protein_rmsd"][source_idx]
            protein_idx += 1
        elif chain == "ligand":
            metrics[f"op_ligand_{label}_rmsd"] = rmsd_result["ligand_rmsd"][ligand_idx]
            metrics[f"op_atom_distances_{label}"] = rmsd_result["atom_distances"][ligand_idx]
            metrics[f"origin_ligand_{label}_rmsd"] = rmsd_origin["ligand_rmsd"][ligand_idx]
            metrics[f"origin_atom_distances_{label}"] = rmsd_origin["atom_distances"][ligand_idx]
            ligand_idx += 1
        elif chain == "dna":
            metrics[f"op_dna_{label}_rmsd"] = rmsd_result["dna_rmsd"]
            metrics[f"origin_dna_{label}_rmsd"] = rmsd_origin["dna_rmsd"]
        elif chain == "rna":
            metrics[f"op_rna_{label}_rmsd"] = rmsd_result["rna_rmsd"]
            metrics[f"origin_rna_{label}_rmsd"] = rmsd_origin["rna_rmsd"]


def _add_interface_eval_metrics(metric, chain_pair_iptm, protein_count, count, chain_labels):
    other_indices = range(protein_count, count)
    values = []
    for i in range(protein_count):
        for j in other_indices:
            values.extend([_matrix_value(chain_pair_iptm, i, j), _matrix_value(chain_pair_iptm, j, i)])
    values = [value for value in values if value is not None]
    metric["eval_all_iptm_to_protein"] = float(np.mean(values)) if values else 0
    metric["eval_all_ipae_to_protein"] = None

    for j in other_indices:
        label = chain_labels[j]
        chain_values = []
        for i in range(protein_count):
            chain_values.extend([_matrix_value(chain_pair_iptm, i, j), _matrix_value(chain_pair_iptm, j, i)])
        chain_values = [value for value in chain_values if value is not None]
        metric[f"eval_{label}_iptm"] = float(np.mean(chain_values)) if chain_values else 0
        metric[f"eval_{label}_ipae"] = None


def _add_interface_op_metrics(metrics, chain_pair_iptm, pae, chain_ids, protein_count, count, chain_labels):
    other_indices = range(protein_count, count)
    iptm_values = []
    ipae_values = []
    for i in range(protein_count):
        for j in other_indices:
            iptm_values.extend([_matrix_value(chain_pair_iptm, i, j), _matrix_value(chain_pair_iptm, j, i)])
            ipae_values.extend([_chain_pair_mean(pae, chain_ids, i, j), _chain_pair_mean(pae, chain_ids, j, i)])
    iptm_values = [value for value in iptm_values if value is not None]
    ipae_values = [value for value in ipae_values if value is not None]
    metrics["op_all_iptm_to_protein"] = float(np.mean(iptm_values)) if iptm_values else 0
    metrics["op_all_ipae_to_protein"] = float(np.mean(ipae_values)) if ipae_values else None

    for j in other_indices:
        label = chain_labels[j]
        chain_iptm = []
        chain_ipae = []
        for i in range(protein_count):
            chain_iptm.extend([_matrix_value(chain_pair_iptm, i, j), _matrix_value(chain_pair_iptm, j, i)])
            chain_ipae.extend([_chain_pair_mean(pae, chain_ids, i, j), _chain_pair_mean(pae, chain_ids, j, i)])
        chain_iptm = [value for value in chain_iptm if value is not None]
        chain_ipae = [value for value in chain_ipae if value is not None]
        metrics[f"op_{label}_iptm"] = float(np.mean(chain_iptm)) if chain_iptm else 0
        metrics[f"op_{label}_ipae"] = float(np.mean(chain_ipae)) if chain_ipae else None


def _chain_pair_mean(matrix, chain_ids, chain_a, chain_b):
    mask = np.outer(chain_ids == chain_a, chain_ids == chain_b)
    if not mask.any():
        return None
    return float(matrix[mask].mean())

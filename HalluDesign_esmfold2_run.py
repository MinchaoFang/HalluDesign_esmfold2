import argparse
import ast
import copy
import os
import csv

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEFAULT_ESMFOLD2_MODEL_PATH = (
    "/storage/caolab/fangmc/cache/huggingface/hub/"
    "models--biohub--ESMFold2/snapshots/"
    "e1e189d0f5fb70c2693da2332eca4443c0ccccd6"
)
DEFAULT_ESMC_MODEL_PATH = (
    "/storage/caolab/fangmc/cache/huggingface/hub/"
    "models--biohub--ESMC-6B/snapshots/"
    "89c554c46a44d825fbfbe3ce2a6bdc539770bdaa"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run HalluDesign optimization with the Biohub ESMFold2 backend."
    )
    parser.add_argument("--pdb_list", type=str, required=False)
    parser.add_argument("--input_file", type=str, required=False)
    parser.add_argument("--fix_res_index", type=str, required=False)
    parser.add_argument("--fix_chain_index", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_seqs", type=int, default=8)
    parser.add_argument("--num_recycles", type=int, default=10)
    parser.add_argument("--ref_time_steps", type=int, default=6)
    parser.add_argument("--cdr", type=str, required=False)
    parser.add_argument("--fix_seq_file", type=str, required=False)
    parser.add_argument("--framework_seq", type=str, nargs="+", required=False, default=[])
    parser.add_argument("--template_path", type=str, required=True)
    parser.add_argument("--template_for_eval", type=str, required=False)
    parser.add_argument("--HalluDesign_model", type=str, default="esmfold2", choices=["esmfold2"])
    parser.add_argument("--sm", type=str, nargs="+", required=False, default=[])
    parser.add_argument("--ccd", type=str, nargs="+", required=False, default=[])
    parser.add_argument("--mpnn", type=str, required=False)
    parser.add_argument("--mpnn_temperature", type=float, default=0.1)
    parser.add_argument("--dna", type=str, nargs="+", required=False, default=[])
    parser.add_argument("--rna", type=str, nargs="+", required=False, default=[])
    parser.add_argument(
        "--design_epoch_begin",
        type=int,
        required=False,
        default=0,
        help=(
            "0-based recycle where multi-seq MPNN evaluation begins; "
            "earlier cycles run single-seq warmup."
        ),
    )
    parser.add_argument("--symmetry_residues", type=str, default="")
    parser.add_argument("--symmetry_chains", type=str, default="")
    parser.add_argument("--symmetry_segments", type=int, default=0)
    parser.add_argument(
        "--cyclic",
        type=int,
        default=0,
        help=(
            "Use cyclic residue-index positional encoding: "
            "1 for the first protein chain, 3 for the first three protein chains."
        ),
    )
    parser.add_argument("--random_init", action="store_true", default=False)

    parser.add_argument("--esmfold2_model_path", type=str, default=DEFAULT_ESMFOLD2_MODEL_PATH)
    parser.add_argument("--esmc_model_path", type=str, default=DEFAULT_ESMC_MODEL_PATH)
    parser.add_argument("--esmfold2_num_loops", type=int, default=3)
    parser.add_argument("--esmfold2_num_sampling_steps", type=int, default=50)
    parser.add_argument(
        "--esmfold2_dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument("--esmfold2_allow_download", action="store_true", default=False)
    parser.add_argument("--esmfold2_chunk_size", type=int, default=64)
    parser.add_argument("--esmfold2_device", type=str, default="auto")
    parser.add_argument("--esmfold2_max_inference_sigma", type=float, default=256.0)
    return parser.parse_args()


def build_mpnn_model(args):
    from LigandMPNN.package import MPNNModel

    mpnn_name = args.mpnn
    if not mpnn_name:
        mpnn_name = "ligand_mpnn" if (args.sm or args.ccd or args.dna or args.rna) else "protein_mpnn"
    args.mpnn = mpnn_name

    common_kwargs = dict(
        T=args.mpnn_temperature,
        ligand_mpnn_use_side_chain_context=1,
        ligand_mpnn_use_atom_context=1,
        number_of_packs_per_design=1,
        pack_side_chains=1,
        parse_atoms_with_zero_occupancy=1,
        pack_with_ligand_context=0,
        repack_everything=1,
    )

    if mpnn_name == "ligandmpnn_plus_proteinmpnn":
        ligand_mpnn_model = MPNNModel(model_name="ligand_mpnn", **common_kwargs)
        protein_mpnn_model = MPNNModel(model_name="soluble_mpnn", **common_kwargs)
        return [ligand_mpnn_model, protein_mpnn_model]

    return MPNNModel(model_name=mpnn_name, **common_kwargs)


def load_pdb_files(args):
    if args.pdb_list and args.input_file:
        raise ValueError("Cannot specify both --pdb_list and --input_file")
    if args.pdb_list:
        with open(args.pdb_list, "r") as handle:
            return [line.strip() for line in handle if line.strip()]
    if args.input_file:
        return [args.input_file]
    raise ValueError("You must specify either --pdb_list or --input_file")


def main():
    args = parse_arguments()

    from data.utility import count_chain_based_on_template_json
    from esmfold2_model import ESMFold2Inferrer
    from eval.eval_utility import generate_metrics
    from filelock import FileLock
    from models_utility import esmfold2_op_eval

    if not os.path.exists(args.template_path):
        raise FileNotFoundError(f"Template file {args.template_path} not found")
    if args.symmetry_residues and args.symmetry_chains:
        raise ValueError("Cannot specify both --symmetry_residues and --symmetry_chains")
    if args.sm and args.ccd:
        raise ValueError("Cannot specify both --sm and --ccd at the same time")

    os.makedirs(args.output_dir, exist_ok=True)
    pdb_files = load_pdb_files(args)
    mpnn_model = build_mpnn_model(args)
    mpnn_config_dict = {
        "temperature": args.mpnn_temperature,
        "model_name": args.mpnn,
        "num_seqs": 1,
    }

    fixed_residues = args.fix_res_index.split() if args.fix_res_index else []
    fixed_chains = args.fix_chain_index.split() if args.fix_chain_index else []

    chunk_size = args.esmfold2_chunk_size if args.esmfold2_chunk_size > 0 else None
    designer_model = ESMFold2Inferrer(
        model_name=args.esmfold2_model_path,
        esmc_model_name=args.esmc_model_path,
        num_loops=args.esmfold2_num_loops,
        num_sampling_steps=args.esmfold2_num_sampling_steps,
        dtype=args.esmfold2_dtype,
        local_files_only=not args.esmfold2_allow_download,
        device=args.esmfold2_device,
        chunk_size=chunk_size,
        max_inference_sigma=args.esmfold2_max_inference_sigma,
        cyclic=args.cyclic,
    )

    protein_chains, ligand_chains, dna_chains, rna_chains, chain_types = (
        count_chain_based_on_template_json(args.template_path)
    )
    metrics_template = generate_metrics(
        protein_chains,
        ligand_chains,
        dna_chains,
        rna_chains,
        chain_types,
    )

    csv_path = os.path.join(args.output_dir, "processing_results.csv")
    lock = FileLock(f"{csv_path}.lock")

    for pdb_file in pdb_files:
        print(f"\nProcessing {pdb_file}...")
        current_input = pdb_file
        chain_number_list_cdr = []
        bais_per_residues = None
        local_fixed_residues = list(fixed_residues)

        if args.fix_seq_file:
            import pandas as pd

            df_fix = pd.read_csv(args.fix_seq_file)
            rows = df_fix[df_fix["file_path"].apply(os.path.basename) == os.path.basename(pdb_file)]
            if "fix_res" in df_fix.columns and not rows.empty:
                local_fixed_residues = str(rows["fix_res"].values[0]).split()
            if "bias" in df_fix.columns and not rows.empty:
                bais_per_residues = ast.literal_eval(rows["bias"].values[0])

        for cycle in range(args.num_recycles):
            print(f"  Starting cycle {cycle + 1}")
            is_last_cycle = cycle == args.num_recycles - 1
            design_begin = cycle >= args.design_epoch_begin
            if design_begin:
                mpnn_config_dict["num_seqs"] = args.num_seqs
            else:
                mpnn_config_dict["num_seqs"] = 1
            print(
                f"begin multi-batch evaluation {design_begin}; "
                f"MPNN num_seqs={mpnn_config_dict['num_seqs']}"
            )

            metrics = copy.deepcopy(metrics_template)
            try:
                metrics, next_input, chain_number_list_cdr = esmfold2_op_eval(
                    pdb_file=current_input,
                    cycle=cycle,
                    output_dir=args.output_dir,
                    template_path=args.template_path,
                    template_for_eval=args.template_for_eval,
                    mpnn_model=mpnn_model,
                    mpnn_config_dict=mpnn_config_dict,
                    designer_model=designer_model,
                    ref_time_steps=args.ref_time_steps,
                    chain_types=chain_types,
                    fixed_chains=fixed_chains,
                    fixed_residues=local_fixed_residues,
                    bais_per_residues=bais_per_residues,
                    metrics=metrics,
                    symmetry_residues=args.symmetry_residues,
                    symmetry_chains=args.symmetry_chains,
                    symmetry_segments=args.symmetry_segments,
                    sm=args.sm,
                    ccd=args.ccd,
                    dna=args.dna,
                    rna=args.rna,
                    cdr=args.cdr,
                    framework_seq=args.framework_seq,
                    design_begin=design_begin,
                    chain_number_list_cdr=chain_number_list_cdr,
                    cyclic=args.cyclic,
                    random_init=args.random_init,
                    run_esmfold2=not is_last_cycle,
                )
                current_input = next_input

                with lock:
                    file_exists = os.path.exists(csv_path)
                    with open(csv_path, "a", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
                        if not file_exists:
                            writer.writeheader()
                        writer.writerows(metrics)

                if is_last_cycle:
                    break
            except Exception as exc:
                print(f"  Error in cycle {cycle + 1}: {exc}")
                continue

    print(f"Processing completed. Results saved to {csv_path}")


if __name__ == "__main__":
    main()

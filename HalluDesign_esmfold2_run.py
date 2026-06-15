import argparse
import ast
import copy
import os
import csv
import json
import random
import shutil
import string
from dataclasses import dataclass

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

PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class DesignJob:
    input_path: str | None
    file_tag: str
    design_index: int
    seed: int


@dataclass(frozen=True)
class TemplateChain:
    chain_id: str
    chain_type: str
    sequence_index: int
    block_key: str
    count: int


def chain_id_stream():
    letters = string.ascii_uppercase
    for letter in letters:
        yield letter
    for first in letters:
        for second in letters:
            yield first + second


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
    parser.add_argument(
        "--fix_seq_file",
        type=str,
        required=False,
        help=(
            "CSV with file_path and optional fix_res/bias columns. file_path can "
            "match an input PDB basename or a no-PDB design tag such as random_init_001."
        ),
    )
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
    parser.add_argument(
        "--random_init_chain_spec",
        type=str,
        default="",
        help=(
            "No-PDB random-init chain length spec, e.g. A:80 or A:50-80. "
            "Only protein chains are redesigned by this ESMFold2 runner."
        ),
    )
    parser.add_argument(
        "--num_designs",
        type=int,
        default=1,
        help=(
            "Number of independent no-PDB random-init design trajectories. "
            "Only used with --random_init_chain_spec and without --input_file/--pdb_list."
        ),
    )
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--esmfold2_model_path", type=str, default=DEFAULT_ESMFOLD2_MODEL_PATH)
    parser.add_argument("--esmc_model_path", type=str, default=DEFAULT_ESMC_MODEL_PATH)
    parser.add_argument("--esmfold2_num_loops", type=int, default=3)
    parser.add_argument("--esmfold2_num_sampling_steps", type=int, default=50)
    parser.add_argument("--esmfold2_num_diffusion_samples", type=int, default=5)
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


def _block_type_and_key(sequence_item):
    if "proteinChain" in sequence_item:
        return "protein", "proteinChain"
    if "dnaSequence" in sequence_item:
        return "dna", "dnaSequence"
    if "rnaSequence" in sequence_item:
        return "rna", "rnaSequence"
    if "ligand" in sequence_item:
        return "ligand", "ligand"
    raise ValueError(f"Unsupported template sequence block: {sequence_item}")


def parse_template_chains(template_path):
    with open(template_path, "r") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError("ESMFold2 template must be a JSON list with one job entry.")

    chains = []
    ids = chain_id_stream()
    for index, item in enumerate(data[0].get("sequences", [])):
        chain_type, block_key = _block_type_and_key(item)
        count = int(item[block_key].get("count", 1))
        for _ in range(count):
            chains.append(
                TemplateChain(
                    chain_id=next(ids),
                    chain_type=chain_type,
                    sequence_index=index,
                    block_key=block_key,
                    count=count,
                )
            )
    return chains


def parse_random_init_chain_spec(spec, template_chains):
    if not spec:
        return {}

    chains_by_id = {chain.chain_id: chain for chain in template_chains}
    ranges = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid --random_init_chain_spec item '{item}'. Use A:80 or A:50-80."
            )
        chain_id, length_spec = item.split(":", 1)
        chain_id = chain_id.strip().upper()
        length_spec = length_spec.strip()
        if chain_id in ranges:
            raise ValueError(f"Duplicate random-init chain spec for chain {chain_id}.")
        if chain_id not in chains_by_id:
            known = ", ".join(chains_by_id)
            raise ValueError(
                f"Chain {chain_id} is not present in the template. Template chains: {known}."
            )

        chain = chains_by_id[chain_id]
        if chain.chain_type != "protein":
            raise ValueError(
                f"Chain {chain_id} is a {chain.chain_type} chain. "
                "HalluDesign_esmfold2 random-init length specs only support protein "
                "chains because the downstream MPNN step redesigns protein sequences."
            )
        if chain.count != 1:
            raise ValueError(
                f"Chain {chain_id} belongs to a template block with count={chain.count}. "
                "For random-init length changes, split this block into count=1 entries."
            )

        if "-" in length_spec:
            low_text, high_text = length_spec.split("-", 1)
            low = int(low_text)
            high = int(high_text)
        else:
            low = high = int(length_spec)
        if low <= 0 or high <= 0 or low > high:
            raise ValueError(
                f"Invalid length range for chain {chain_id}: {length_spec}."
            )
        ranges[chain_id] = (low, high)
    return ranges


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


def load_design_jobs(args):
    if args.num_designs < 1:
        raise ValueError("--num_designs must be at least 1")
    if args.pdb_list and args.input_file:
        raise ValueError("Cannot specify both --pdb_list and --input_file")
    if args.num_designs != 1 and (args.input_file or args.pdb_list):
        raise ValueError("--num_designs is only supported for no-PDB random init")
    if args.random_init_chain_spec and (args.input_file or args.pdb_list):
        raise ValueError(
            "--random_init_chain_spec is for no-PDB random init; do not combine it "
            "with --input_file or --pdb_list"
        )
    if args.pdb_list:
        with open(args.pdb_list, "r") as handle:
            pdb_files = [
                os.path.abspath(os.path.expanduser(line.strip()))
                for line in handle
                if line.strip()
            ]
        return [
            DesignJob(
                input_path=pdb_file,
                file_tag=os.path.splitext(os.path.basename(pdb_file))[0].lower(),
                design_index=index,
                seed=args.seed,
            )
            for index, pdb_file in enumerate(pdb_files)
        ]
    if args.input_file:
        return [
            DesignJob(
                input_path=args.input_file,
                file_tag=os.path.splitext(os.path.basename(args.input_file))[0].lower(),
                design_index=0,
                seed=args.seed,
            )
        ]
    if args.random_init_chain_spec:
        args.random_init = True
        return [
            DesignJob(
                input_path=None,
                file_tag=f"random_init_{index + 1:03d}",
                design_index=index,
                seed=args.seed + index,
            )
            for index in range(args.num_designs)
        ]
    raise ValueError("You must specify --input_file, --pdb_list, or --random_init_chain_spec")


def load_fix_seq_table(fix_seq_file):
    if not fix_seq_file:
        return None
    import pandas as pd

    table = pd.read_csv(fix_seq_file)
    if "file_path" not in table.columns:
        raise ValueError("--fix_seq_file must contain a file_path column")
    return table


def _cell_has_value(value):
    if value is None:
        return False
    try:
        import pandas as pd

        if pd.isna(value):
            return False
    except Exception:
        try:
            if value != value:
                return False
        except Exception:
            pass
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null"}


def _fix_seq_keys(value):
    if not _cell_has_value(value):
        return set()
    text = str(value).strip()
    base = os.path.basename(text)
    stem = os.path.splitext(base)[0]
    return {text, base, stem}


def _job_fix_seq_keys(job):
    keys = {
        job.file_tag,
        f"{job.file_tag}.pdb",
        f"{job.file_tag}.cif",
        f"{job.file_tag}_recycle_1",
        f"{job.file_tag}_recycle_1.pdb",
        f"{job.file_tag}_recycle_1.cif",
    }
    if job.input_path:
        keys.update(_fix_seq_keys(job.input_path))
    return keys


def fix_seq_settings_for_job(fix_seq_table, job, default_fixed_residues):
    fixed_residues = list(default_fixed_residues)
    bais_per_residues = None
    if fix_seq_table is None:
        return fixed_residues, bais_per_residues

    candidate_keys = _job_fix_seq_keys(job)
    matched_row = None
    for _, row in fix_seq_table.iterrows():
        if candidate_keys & _fix_seq_keys(row["file_path"]):
            matched_row = row
            break

    if matched_row is None:
        print(
            "No --fix_seq_file row matched "
            f"{job.file_tag}; using command-line fixed residues and no bias."
        )
        return fixed_residues, bais_per_residues

    print(f"--fix_seq_file matched {matched_row['file_path']} for {job.file_tag}")
    if "fix_res" in fix_seq_table.columns:
        if _cell_has_value(matched_row.get("fix_res")):
            fixed_residues = str(matched_row["fix_res"]).split()
        else:
            fixed_residues = []

    if "bias" in fix_seq_table.columns and _cell_has_value(matched_row.get("bias")):
        try:
            bais_per_residues = ast.literal_eval(str(matched_row["bias"]))
        except Exception as exc:
            raise ValueError(
                f"Could not parse bias column for {matched_row['file_path']}: "
                f"{matched_row['bias']}"
            ) from exc

    return fixed_residues, bais_per_residues


def _random_sequence(chain_type, length, rng):
    if chain_type != "protein":
        raise ValueError(f"Cannot randomize chain type {chain_type}")
    alphabet = PROTEIN_ALPHABET
    return "".join(rng.choice(alphabet) for _ in range(length))


def write_random_init_json(template_path, output_dir, tag, template_chains, chain_ranges, seed):
    rng = random.Random(seed)
    with open(template_path, "r") as handle:
        input_json = copy.deepcopy(json.load(handle))
    input_json[0]["name"] = tag

    changed = []
    for chain in template_chains:
        if chain.chain_id not in chain_ranges:
            continue
        low, high = chain_ranges[chain.chain_id]
        length = rng.randint(low, high)
        sequence = _random_sequence(chain.chain_type, length, rng)
        input_json[0]["sequences"][chain.sequence_index][chain.block_key]["sequence"] = sequence
        changed.append(f"{chain.chain_id}:{chain.chain_type}:{length}")

    if not changed:
        raise ValueError("No chains were randomized for no-PDB initialization.")

    json_path = os.path.join(output_dir, f"{tag}.json")
    with open(json_path, "w") as handle:
        json.dump(input_json, handle, indent=2)
    print(f"Random-init template {tag}: {', '.join(changed)}")
    return json_path, input_json


def apply_sequence_overrides(input_json, chain_types, sm, ccd, dna, rna):
    sm_count = 0
    dna_count = 0
    rna_count = 0
    for index, chain_type in enumerate(chain_types):
        if chain_type == "ligand":
            ligand_block = input_json[0]["sequences"][index]["ligand"]
            if sm_count < len(ccd):
                ligand_block.pop("ligand", None)
                ligand_block.pop("smiles", None)
                ligand_block["ccdCodes"] = [ccd[sm_count]]
            elif sm_count < len(sm):
                ligand_block.pop("ccd", None)
                ligand_block.pop("ccdCodes", None)
                ligand_block["ligand"] = sm[sm_count]
            sm_count += 1
        elif chain_type == "dna":
            if dna_count < len(dna):
                input_json[0]["sequences"][index]["dnaSequence"]["sequence"] = dna[dna_count]
            dna_count += 1
        elif chain_type == "rna":
            if rna_count < len(rna):
                input_json[0]["sequences"][index]["rnaSequence"]["sequence"] = rna[rna_count]
            rna_count += 1


def protein_sequences_from_template_json(input_json):
    sequences = []
    for item in input_json[0].get("sequences", []):
        if "proteinChain" in item:
            sequences.append(item["proteinChain"].get("sequence", ""))
    return ":".join(sequences) if sequences else None


def count_tuple_from_chain_types(chain_types):
    return [
        len(chain_types),
        chain_types.count("protein"),
        chain_types.count("ligand"),
        chain_types.count("rna"),
        chain_types.count("dna"),
    ]


def annotate_metrics(metrics, job):
    if isinstance(metrics, dict):
        metrics = [metrics]
    for metric in metrics:
        metric["design_index"] = job.design_index
        metric["design_tag"] = job.file_tag
        metric["seed"] = job.seed
    return metrics


def append_metrics(csv_path, lock, metrics):
    with lock:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerows(metrics)


def run_no_pdb_random_init_cycle(
    job,
    cycle,
    args,
    designer_model,
    metrics_template,
    template_chains,
    random_init_chain_ranges,
    chain_types,
):
    from data.utility import convert_cif_to_pdb
    from eval.evaluation import process_confidence_metrics_esmfold2

    backend_name = getattr(designer_model, "backend_name", "ESMFold2")
    target_dir = os.path.join(args.output_dir, f"recycle_{cycle + 1}")
    os.makedirs(target_dir, exist_ok=True)
    tag = f"{job.file_tag}_recycle_{cycle + 1}"
    json_path, input_json = write_random_init_json(
        args.template_path,
        target_dir,
        tag,
        template_chains,
        random_init_chain_ranges,
        job.seed,
    )
    apply_sequence_overrides(input_json, chain_types, args.sm, args.ccd, args.dna, args.rna)
    with open(json_path, "w") as handle:
        json.dump(input_json, handle, indent=2)

    print(f"begin {backend_name} no-PDB random-init prediction")
    results_op = designer_model.predict(
        input_json_path=json_path,
        dump_dir=target_dir,
        seed=job.seed,
    )

    metrics = copy.deepcopy(metrics_template)
    metrics["file_name"] = job.file_tag
    metrics["cycle"] = cycle
    metrics["mpnn_model"] = "random_init"
    metrics["mpnn_sequence"] = protein_sequences_from_template_json(input_json)

    if not results_op:
        metrics["HalluDesign_Status"] = f"{backend_name}_random_init_failed"
        return [metrics], None

    tag_for_path = tag.lower()
    cif_path = os.path.join(
        target_dir,
        tag_for_path,
        f"seed_{job.seed}",
        "predictions",
        f"{tag_for_path}_seed_{job.seed}_sample_0.cif",
    )
    pdb_output = cif_path.replace(".cif", ".pdb")
    next_input = None
    if convert_cif_to_pdb(cif_path, pdb_output):
        canonical_pdb = os.path.join(target_dir, f"{tag_for_path}.pdb")
        shutil.copy(pdb_output, canonical_pdb)
        next_input = canonical_pdb
        metrics["origin_path"] = canonical_pdb
        metrics["packed_path"] = canonical_pdb
        metrics = process_confidence_metrics_esmfold2(
            results_op,
            cif_path,
            canonical_pdb,
            canonical_pdb,
            [metrics],
            [],
            chain_types,
            [],
            count_tuple_from_chain_types(chain_types),
        )
    else:
        metrics["op_cif_path"] = cif_path
        metrics["HalluDesign_Status"] = f"{backend_name}_random_init_convert_failed"
        metrics = [metrics]
    return metrics, next_input


def main():
    args = parse_arguments()
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    args.template_path = os.path.abspath(os.path.expanduser(args.template_path))
    if args.input_file:
        args.input_file = os.path.abspath(os.path.expanduser(args.input_file))
    if args.pdb_list:
        args.pdb_list = os.path.abspath(os.path.expanduser(args.pdb_list))
    if args.template_for_eval:
        args.template_for_eval = os.path.abspath(os.path.expanduser(args.template_for_eval))
    if args.fix_seq_file:
        args.fix_seq_file = os.path.abspath(os.path.expanduser(args.fix_seq_file))

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
    template_chains = parse_template_chains(args.template_path)
    random_init_chain_ranges = parse_random_init_chain_spec(
        args.random_init_chain_spec,
        template_chains,
    )
    design_jobs = load_design_jobs(args)
    mpnn_model = build_mpnn_model(args)
    mpnn_config_dict = {
        "temperature": args.mpnn_temperature,
        "model_name": args.mpnn,
        "num_seqs": 1,
    }

    fixed_residues = args.fix_res_index.split() if args.fix_res_index else []
    fixed_chains = args.fix_chain_index.split() if args.fix_chain_index else []
    fix_seq_table = load_fix_seq_table(args.fix_seq_file)

    chunk_size = args.esmfold2_chunk_size if args.esmfold2_chunk_size > 0 else None
    designer_model = ESMFold2Inferrer(
        model_name=args.esmfold2_model_path,
        esmc_model_name=args.esmc_model_path,
        num_loops=args.esmfold2_num_loops,
        num_sampling_steps=args.esmfold2_num_sampling_steps,
        num_diffusion_samples=args.esmfold2_num_diffusion_samples,
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

    for job in design_jobs:
        print(
            "\nProcessing "
            f"{job.input_path if job.input_path else 'no-PDB random initialization'} "
            f"as {job.file_tag} with seed {job.seed}..."
        )
        current_input = job.input_path
        chain_number_list_cdr = []
        local_fixed_residues, bais_per_residues = fix_seq_settings_for_job(
            fix_seq_table,
            job,
            fixed_residues,
        )

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
                if current_input is None:
                    metrics, next_input = run_no_pdb_random_init_cycle(
                        job=job,
                        cycle=cycle,
                        args=args,
                        designer_model=designer_model,
                        metrics_template=metrics_template,
                        template_chains=template_chains,
                        random_init_chain_ranges=random_init_chain_ranges,
                        chain_types=chain_types,
                    )
                else:
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
                        seed=job.seed,
                        run_esmfold2=not is_last_cycle,
                    )
                current_input = next_input
                metrics = annotate_metrics(metrics, job)
                append_metrics(csv_path, lock, metrics)

                if is_last_cycle or current_input is None:
                    break
            except Exception as exc:
                print(f"  Error in cycle {cycle + 1}: {exc}")
                continue

    print(f"Processing completed. Results saved to {csv_path}")


if __name__ == "__main__":
    main()

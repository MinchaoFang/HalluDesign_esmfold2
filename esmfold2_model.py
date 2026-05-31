from __future__ import annotations

import json
import os
import string
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


class ESMFold2DependencyError(ImportError):
    pass


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

_ATOMIC_NUM_TO_ELEMENT = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "CL",
    35: "BR",
    53: "I",
}


def _chain_id_stream() -> Iterable[str]:
    letters = string.ascii_uppercase
    for letter in letters:
        yield letter
    for first in letters:
        for second in letters:
            yield first + second


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _decode_atom_name(encoded: torch.Tensor) -> str:
    chars = []
    for item in encoded.tolist():
        if item:
            chars.append(chr(int(item) + 32))
    return "".join(chars).strip()


def _element_symbol(atomic_num: torch.Tensor | int) -> str:
    if hasattr(atomic_num, "item"):
        atomic_num = int(atomic_num.item())
    return _ATOMIC_NUM_TO_ELEMENT.get(int(atomic_num), "")


def _torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = dtype_name.lower()
    if dtype_name in ("float32", "fp32"):
        return torch.float32
    if dtype_name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dtype_name in ("float16", "fp16"):
        return torch.float16
    raise ValueError(f"Unsupported ESMFold2 dtype: {dtype_name}")


def _relative_position_forward_with_halludesign_cyclic(
    self,
    residue_index: torch.Tensor,
    asym_id: torch.Tensor,
    sym_id: torch.Tensor,
    entity_id: torch.Tensor,
    token_index: torch.Tensor,
) -> torch.Tensor:
    """ESMFold2 relative position encoding with cyclic residue-index distances."""
    bij_same_chain = asym_id.unsqueeze(2) == asym_id.unsqueeze(1)
    bij_same_residue = residue_index.unsqueeze(2) == residue_index.unsqueeze(1)
    bij_same_entity = entity_id.unsqueeze(2) == entity_id.unsqueeze(1)

    raw_dij_residue = residue_index.unsqueeze(2) - residue_index.unsqueeze(1)
    cyclic_dij_residue = raw_dij_residue
    target_lengths = getattr(self, "_hallu_cyclic_asym_lengths", {})

    for target_asym_id, chain_length in target_lengths.items():
        chain_length = int(chain_length)
        if chain_length <= 1:
            continue
        chain_mask = asym_id == int(target_asym_id)
        residue_mask = residue_index < chain_length
        pair_mask = (
            chain_mask.unsqueeze(2)
            & chain_mask.unsqueeze(1)
            & residue_mask.unsqueeze(2)
            & residue_mask.unsqueeze(1)
        )
        wrapped = raw_dij_residue
        half_length = chain_length // 2
        wrapped = torch.where(
            wrapped > half_length,
            wrapped - chain_length,
            wrapped,
        )
        wrapped = torch.where(
            wrapped < -half_length,
            wrapped + chain_length,
            wrapped,
        )
        cyclic_dij_residue = torch.where(pair_mask, wrapped, cyclic_dij_residue)

    dij_residue = torch.clip(
        cyclic_dij_residue + self.n_relative_residx_bins,
        0,
        2 * self.n_relative_residx_bins,
    )
    dij_residue = torch.where(
        bij_same_chain, dij_residue, 2 * self.n_relative_residx_bins + 1
    )
    aij_rel_pos = F.one_hot(dij_residue, 2 * self.n_relative_residx_bins + 2)

    dij_token = torch.clip(
        token_index.unsqueeze(2)
        - token_index.unsqueeze(1)
        + self.n_relative_residx_bins,
        0,
        2 * self.n_relative_residx_bins,
    )
    dij_token = torch.where(
        bij_same_chain & bij_same_residue,
        dij_token,
        2 * self.n_relative_residx_bins + 1,
    )
    aij_rel_token = F.one_hot(dij_token, 2 * self.n_relative_residx_bins + 2)

    dij_chain = torch.clip(
        sym_id.unsqueeze(2) - sym_id.unsqueeze(1) + self.n_relative_chain_bins,
        0,
        2 * self.n_relative_chain_bins,
    )
    dij_chain = torch.where(
        bij_same_chain, 2 * self.n_relative_chain_bins + 1, dij_chain
    )
    aij_rel_chain = F.one_hot(dij_chain, 2 * self.n_relative_chain_bins + 2)

    feats = torch.cat(
        [
            aij_rel_pos.float(),
            aij_rel_token.float(),
            bij_same_entity.float().unsqueeze(-1),
            aij_rel_chain.float(),
        ],
        dim=-1,
    )
    return self.embed(feats)


@contextmanager
def _torch_seed(seed: int | None):
    if seed is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@torch.inference_mode()
def _sample_with_halludesign_refinement(
    self,
    z_trunk: torch.Tensor,
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor | None,
    relative_position_encoding: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_charge: torch.Tensor,
    ref_mask: torch.Tensor,
    ref_element: torch.Tensor,
    ref_atom_name_chars: torch.Tensor,
    ref_space_uid: torch.Tensor,
    tok_idx: torch.Tensor,
    asym_id: torch.Tensor,
    residue_index: torch.Tensor,
    entity_id: torch.Tensor,
    token_index: torch.Tensor,
    sym_id: torch.Tensor,
    token_attention_mask: torch.Tensor | None = None,
    num_diffusion_samples: int = 1,
    num_sampling_steps: int | None = None,
    max_inference_sigma: float | None = 256.0,
    noise_scale: float | None = None,
    step_scale: float | None = None,
    return_atom_repr: bool = False,
    use_inference_cache: bool = True,
    denoising_early_exit_rmsd: float | None = None,
) -> dict[str, torch.Tensor | None]:
    """ESMFold2 sampler with optional HalluDesign coordinate initialization."""
    n_atoms = tok_idx.shape[1]
    device = s_inputs.device
    target_batch = s_inputs.shape[0] * num_diffusion_samples

    inference_cache: dict[str, torch.Tensor] | None = (
        {} if use_inference_cache else None
    )

    steps = self.inference_num_steps if num_sampling_steps is None else int(num_sampling_steps)
    schedule = self.inference_noise_schedule(steps, device)

    max_sigma_override = getattr(self, "_hallu_max_inference_sigma", max_inference_sigma)
    if max_sigma_override is not None:
        schedule = schedule[schedule <= float(max_sigma_override)]
        schedule = F.pad(schedule, (1, 0), value=float(max_sigma_override))

    lam = (
        self.noise_scale
        if noise_scale is None
        else float(noise_scale)
    )
    if getattr(self, "_hallu_noise_scale", None) is not None:
        lam = float(self._hallu_noise_scale)

    eta = self.step_scale if step_scale is None else float(step_scale)
    if getattr(self, "_hallu_step_scale", None) is not None:
        eta = float(self._hallu_step_scale)

    atom_mask = ref_mask.repeat_interleave(num_diffusion_samples, 0).float()
    init_coords = getattr(self, "_hallu_init_coords", None)
    fixed_atom_mask = getattr(self, "_hallu_fixed_atom_mask", None)
    refinement_steps = getattr(self, "_hallu_refinement_steps", None)
    if fixed_atom_mask is not None:
        fixed_atom_mask = fixed_atom_mask.to(device=device, dtype=torch.bool)
        if fixed_atom_mask.shape[0] != n_atoms or not bool(fixed_atom_mask.any()):
            fixed_atom_mask = None

    full_num_steps = max(0, len(schedule) - 1)
    use_refinement = (
        init_coords is not None
        and refinement_steps is not None
        and full_num_steps > 0
        and int(refinement_steps) < full_num_steps
    )

    if init_coords is not None and refinement_steps is not None and int(refinement_steps) <= 0:
        x = init_coords.to(device=device, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[0] == 1 and target_batch > 1:
            x = x.repeat_interleave(target_batch, 0)
        return {"sample_atom_coords": x, "diff_token_repr": None}

    if use_refinement:
        remaining_steps = max(1, int(refinement_steps))
        start_idx = full_num_steps - remaining_steps
        schedule = schedule[start_idx:]
        x = init_coords.to(device=device, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[0] == 1 and target_batch > 1:
            x = x.repeat_interleave(target_batch, 0)
        fixed_reference = x[:, fixed_atom_mask, :].clone() if fixed_atom_mask is not None else None
        sigma0 = schedule[0]
        x = x + sigma0 * torch.randn_like(x)
        if fixed_reference is not None:
            x[:, fixed_atom_mask, :] = fixed_reference
    else:
        x = schedule[0] * torch.randn(
            target_batch, n_atoms, 3, device=device, dtype=torch.float32
        )

    gammas = torch.where(
        schedule > self.gamma_min,
        torch.full_like(schedule, self.gamma_0),
        torch.zeros_like(schedule),
    )

    x_denoised_prev: torch.Tensor | None = None
    token_repr: torch.Tensor | None = None
    diff_atom_intermediates: torch.Tensor | None = None

    step_pairs = list(zip(schedule[:-1], schedule[1:], gammas[1:]))
    num_steps = len(step_pairs)

    for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(step_pairs):
        x, x_denoised_prev = self._center_random_augmentation(
            x, atom_mask, second_coords=x_denoised_prev
        )
        fixed_reference = None
        if fixed_atom_mask is not None:
            fixed_reference = x[:, fixed_atom_mask, :].clone()
            if x_denoised_prev is not None:
                x_denoised_prev[:, fixed_atom_mask, :] = fixed_reference

        sigma_tm_val = float(sigma_tm.item())
        t_hat_val = sigma_tm_val * (1.0 + float(gamma.item()))
        eps_std = lam * max(t_hat_val**2 - sigma_tm_val**2, 0.0) ** 0.5
        x_noisy = x + eps_std * torch.randn_like(x)
        if fixed_reference is not None:
            x_noisy[:, fixed_atom_mask, :] = fixed_reference

        is_last_step = step_idx == num_steps - 1
        request_atom_repr = return_atom_repr and (
            is_last_step or denoising_early_exit_rmsd is not None
        )

        dm_out = self.diffusion_module(
            x_noisy=x_noisy,
            t_hat=torch.full(
                (target_batch,), t_hat_val, device=device, dtype=torch.float32
            ),
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_mask=ref_mask,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            ref_space_uid=ref_space_uid,
            tok_idx=tok_idx,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            relative_position_encoding=relative_position_encoding,
            asym_id=asym_id,
            residue_index=residue_index,
            entity_id=entity_id,
            token_index=token_index,
            sym_id=sym_id,
            token_attention_mask=token_attention_mask,
            num_diffusion_samples=num_diffusion_samples,
            return_token_repr=True,
            return_atom_repr=request_atom_repr,
            inference_cache=inference_cache,
        )

        x_denoised = dm_out["x_denoised"]
        if fixed_reference is not None:
            x_denoised[:, fixed_atom_mask, :] = fixed_reference
        token_repr = dm_out["token_repr"]
        if request_atom_repr:
            diff_atom_intermediates = dm_out.get("atom_intermediates")

        with torch.autocast(device_type="cuda", enabled=False):
            x_noisy = self._weighted_rigid_align(
                x_noisy.float(), x_denoised.float(), atom_mask, atom_mask
            )
        x_noisy = x_noisy.to(dtype=x_denoised.dtype)

        sigma_t_val = float(sigma_t.item())
        denoised_over_sigma = (x_noisy - x_denoised) / t_hat_val
        x = x_noisy + eta * (sigma_t_val - t_hat_val) * denoised_over_sigma
        if fixed_reference is not None:
            x[:, fixed_atom_mask, :] = fixed_reference

        if (
            denoising_early_exit_rmsd is not None
            and x_denoised_prev is not None
            and step_idx >= 1
        ):
            with torch.autocast(device_type="cuda", enabled=False):
                aligned = self._weighted_rigid_align(
                    x_denoised_prev.float(),
                    x_denoised.float(),
                    atom_mask,
                    atom_mask,
                )
            diff = (x_denoised.float() - aligned) * atom_mask.unsqueeze(-1)
            per_sample_rmsd = (
                diff.pow(2).sum(dim=(-1, -2)) / atom_mask.sum(dim=-1).clamp(min=1)
            ).sqrt()
            if per_sample_rmsd.max().item() < denoising_early_exit_rmsd:
                x = x_denoised
                x_denoised_prev = x_denoised
                break

        x_denoised_prev = x_denoised

    result: dict[str, torch.Tensor | None] = {
        "sample_atom_coords": x,
        "diff_token_repr": token_repr,
    }
    if return_atom_repr:
        result["diff_atom_intermediates"] = diff_atom_intermediates
    return result


class ESMFold2Inferrer:
    """HalluDesign-compatible wrapper around Biohub ESMFold2."""

    def __init__(
        self,
        model_name: str = DEFAULT_ESMFOLD2_MODEL_PATH,
        esmc_model_name: str = DEFAULT_ESMC_MODEL_PATH,
        num_loops: int = 3,
        num_sampling_steps: int = 0,
        num_diffusion_samples: int = 1,
        dtype: str = "float32",
        local_files_only: bool = True,
        device: str = "auto",
        chunk_size: int | None = None,
        max_inference_sigma: float | None = 256.0,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        cyclic: int = 0,
    ) -> None:
        self.model_name = model_name
        self.esmc_model_name = esmc_model_name
        self.backend_name = "ESMFold2"
        self.num_loops = num_loops
        self.num_sampling_steps = num_sampling_steps if num_sampling_steps > 0 else None
        if num_diffusion_samples != 1:
            warnings.warn(
                "HalluDesign metrics currently expect sample_0; using one ESMFold2 diffusion sample."
            )
        self.num_diffusion_samples = 1
        self.dtype = _torch_dtype(dtype)
        self.local_files_only = local_files_only
        self.max_inference_sigma = max_inference_sigma
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.cyclic = int(cyclic or 0)

        try:
            from esm.models.esmfold2 import (
                DNAInput,
                ESMFold2InputBuilder,
                LigandInput,
                Modification,
                ProteinInput,
                RNAInput,
                StructurePredictionInput,
            )
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
            from transformers.models.esmc.modeling_esmc import ESMCModel
        except Exception as exc:
            raise ESMFold2DependencyError(
                "Biohub ESMFold2 is not available. Install it in this environment with "
                "pip install -r requirements.txt."
            ) from exc

        self.DNAInput = DNAInput
        self.ESMFold2InputBuilder = ESMFold2InputBuilder
        self.LigandInput = LigandInput
        self.Modification = Modification
        self.ProteinInput = ProteinInput
        self.RNAInput = RNAInput
        self.StructurePredictionInput = StructurePredictionInput

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Initializing ESMFold2 model from {model_name} on {device}...")
        self.model = ESMFold2Model.from_pretrained(
            model_name,
            load_esmc=False,
            local_files_only=local_files_only,
        )
        print(f"Initializing ESMC model from {esmc_model_name}...")
        esmc = ESMCModel.from_pretrained(
            esmc_model_name,
            local_files_only=local_files_only,
        )
        self.model._esmc = esmc
        object.__setattr__(self.model, "esmc", self.model._esmc)
        self.model._esmc_fp8 = False
        self.model = self.model.to(self.dtype)
        self.model._esmc = self.model._esmc.to(self.dtype)
        object.__setattr__(self.model, "esmc", self.model._esmc)
        for param in self.model._esmc.parameters():
            param.requires_grad_(False)
        self.model = self.model.to(device).eval()
        if chunk_size is not None:
            self.model.set_chunk_size(chunk_size)
        ccd_cache = self._ccd_cache_dir(model_name)
        self.input_builder = ESMFold2InputBuilder(ccd_cache=ccd_cache)
        self.full_diffusion_steps = self._effective_diffusion_steps()
        print(f"Effective ESMFold2 denoising steps: {self.full_diffusion_steps}")
        if self.cyclic:
            print(
                "ESMFold2 cyclic residue-index positional encoding enabled "
                f"(mode {self.cyclic})."
            )
        print("ESMFold2Inferrer initialized.")

    def predict(
        self,
        input_json_path: str,
        dump_dir: str,
        seed: int,
        input_atom_array_path: str = "",
        diffusion_steps: int = 0,
    ) -> dict[str, Any]:
        spi, sample_name = self._load_structure_prediction_input(input_json_path)
        tag = sample_name.lower()
        output_dir = os.path.join(dump_dir, tag, f"seed_{seed}", "predictions")
        os.makedirs(output_dir, exist_ok=True)

        features, chain_infos = self.input_builder.prepare_input(
            spi, seed=seed, device=self.model.device
        )
        diffusion_steps = int(diffusion_steps or 0)
        configured_sampling_steps = self.num_sampling_steps
        if configured_sampling_steps is None:
            configured_sampling_steps = getattr(
                self.model.structure_head, "inference_num_steps", 0
            )
        if (
            input_atom_array_path
            and configured_sampling_steps
            and diffusion_steps >= int(configured_sampling_steps)
        ):
            print(
                "ESMFold2 full prediction from sequence/SMILES because "
                f"ref_time_steps={diffusion_steps} >= "
                f"esmfold2_num_sampling_steps={int(configured_sampling_steps)}; "
                "ignoring input structure."
            )
            input_atom_array_path = ""
            diffusion_steps = 0

        init_coords = None
        fixed_atom_mask = None
        loaded_coords = self._load_initial_coords(
            input_atom_array_path, features, chain_infos
        )
        if loaded_coords is not None:
            init_coords, fixed_atom_mask = loaded_coords

        diffusion_steps = int(diffusion_steps or 0)
        if init_coords is not None and input_atom_array_path:
            diffusion_steps = self._normalize_refinement_steps(diffusion_steps)
        elif init_coords is None and input_atom_array_path and diffusion_steps:
            print(
                "ESMFold2 refinement requested, but initial coordinates could not be mapped; "
                "falling back to pure ESMFold2 prediction."
            )

        cyclic_asym_lengths = self._cyclic_asym_lengths(chain_infos)
        output = self._run_model(
            features,
            init_coords,
            fixed_atom_mask,
            diffusion_steps,
            seed,
            cyclic_asym_lengths,
        )
        decoded = self.input_builder.decode(
            output,
            features,
            chain_infos,
            num_diffusion_samples=self.num_diffusion_samples,
            complex_id=tag,
        )
        result = decoded[0] if isinstance(decoded, list) else decoded

        cif_path = os.path.join(output_dir, f"{tag}_seed_{seed}_sample_0.cif")
        with open(cif_path, "w") as handle:
            handle.write(result.complex.to_mmcif())

        return self._format_halludesign_result(result, features)

    def _run_model(
        self,
        features: dict[str, Any],
        init_coords: torch.Tensor | None,
        fixed_atom_mask: torch.Tensor | None,
        diffusion_steps: int,
        seed: int,
        cyclic_asym_lengths: dict[int, int] | None,
    ) -> dict[str, torch.Tensor]:
        head = self.model.structure_head
        old_sample = head.sample
        rel_pos = self.model.rel_pos
        old_rel_pos_forward = rel_pos.forward
        head.sample = types.MethodType(_sample_with_halludesign_refinement, head)
        head._hallu_init_coords = init_coords
        head._hallu_fixed_atom_mask = fixed_atom_mask
        head._hallu_refinement_steps = int(diffusion_steps)
        head._hallu_max_inference_sigma = self.max_inference_sigma
        head._hallu_noise_scale = self.noise_scale
        head._hallu_step_scale = self.step_scale
        if cyclic_asym_lengths:
            rel_pos.forward = types.MethodType(
                _relative_position_forward_with_halludesign_cyclic,
                rel_pos,
            )
            rel_pos._hallu_cyclic_asym_lengths = cyclic_asym_lengths
        try:
            with _torch_seed(seed), torch.no_grad():
                return self.model(
                    **features,
                    num_loops=self.num_loops,
                    num_sampling_steps=self.num_sampling_steps,
                    num_diffusion_samples=self.num_diffusion_samples,
                )
        finally:
            head.sample = old_sample
            rel_pos.forward = old_rel_pos_forward
            if hasattr(rel_pos, "_hallu_cyclic_asym_lengths"):
                delattr(rel_pos, "_hallu_cyclic_asym_lengths")
            for attr in (
                "_hallu_init_coords",
                "_hallu_fixed_atom_mask",
                "_hallu_refinement_steps",
                "_hallu_max_inference_sigma",
                "_hallu_noise_scale",
                "_hallu_step_scale",
            ):
                if hasattr(head, attr):
                    delattr(head, attr)

    def _effective_diffusion_steps(self) -> int:
        schedule = self.model.structure_head.inference_noise_schedule(
            self.num_sampling_steps,
            torch.device("cpu"),
        )
        if self.max_inference_sigma is not None:
            schedule = schedule[schedule <= float(self.max_inference_sigma)]
            schedule = F.pad(schedule, (1, 0), value=float(self.max_inference_sigma))
        return max(0, int(schedule.shape[0]) - 1)

    def _normalize_refinement_steps(self, requested_steps: int) -> int:
        requested_steps = max(0, int(requested_steps))
        if requested_steps == 0:
            return 0

        max_refinement_steps = max(0, int(self.full_diffusion_steps) - 1)
        if max_refinement_steps <= 0:
            print(
                "ESMFold2 coordinate refinement requested, but the effective denoising "
                "schedule has no refinement steps; using the input coordinates."
            )
            return 0

        if requested_steps > max_refinement_steps:
            print(
                "ESMFold2 coordinate refinement requested "
                f"{requested_steps} steps; effective full denoising schedule has "
                f"{self.full_diffusion_steps} steps, so clamping to "
                f"{max_refinement_steps} refinement steps."
            )
            return max_refinement_steps

        print(f"ESMFold2 coordinate refinement using {requested_steps} denoising steps.")
        return requested_steps

    def _cyclic_asym_lengths(self, chain_infos: list[Any]) -> dict[int, int]:
        if not self.cyclic:
            return {}

        protein_chains = [
            chain
            for chain in chain_infos
            if int(getattr(chain, "mol_type", -1)) == 0
        ]
        if not protein_chains:
            return {}

        if self.cyclic == 1:
            selected_chains = protein_chains[:1]
        elif self.cyclic == 3:
            selected_chains = protein_chains[:3]
        else:
            selected_chains = protein_chains[:1]
            warnings.warn(
                f"Unsupported ESMFold2 cyclic mode {self.cyclic}; applying mode 1 "
                "to the first protein chain."
            )

        asym_lengths = {}
        chain_labels = []
        for chain in selected_chains:
            residue_indices = [
                int(token.residue_index)
                for token in getattr(chain, "tokens", [])
                if int(getattr(token, "mol_type", -1)) == 0
            ]
            if not residue_indices:
                continue
            chain_length = max(residue_indices) + 1
            asym_id = int(chain.asym_id)
            asym_lengths[asym_id] = chain_length
            chain_labels.append(
                f"{chain.chain_id}(asym_id={asym_id}, length={chain_length})"
            )

        if chain_labels:
            print(
                "Using ESMFold2 cyclic residue-index encoding for "
                + ", ".join(chain_labels)
            )
        return asym_lengths

    @staticmethod
    def _ccd_cache_dir(model_name: str) -> Path | None:
        model_path = Path(model_name)
        if model_path.is_dir() and (model_path / "ccd.pkl").exists():
            return model_path
        return None

    def _load_structure_prediction_input(self, json_path: str):
        with open(json_path, "r") as handle:
            data = json.load(handle)
        if not isinstance(data, list) or not data:
            raise ValueError(
                "HalluDesign_esmfold2 expects a template JSON list with a 'sequences' block."
            )
        job = data[0]
        return self._from_template_json(job), job.get("name", "pred")

    def _from_template_json(self, job: dict[str, Any]):
        chain_ids = _chain_id_stream()
        sequences = []
        for item in job.get("sequences", []):
            if "proteinChain" in item:
                block = item["proteinChain"]
                ids = [next(chain_ids) for _ in range(int(block.get("count", 1)))]
                sequences.append(
                    self.ProteinInput(
                        id=ids if len(ids) > 1 else ids[0],
                        sequence=block["sequence"],
                    )
                )
            elif "dnaSequence" in item:
                block = item["dnaSequence"]
                ids = [next(chain_ids) for _ in range(int(block.get("count", 1)))]
                sequences.append(
                    self.DNAInput(
                        id=ids if len(ids) > 1 else ids[0],
                        sequence=block["sequence"],
                    )
                )
            elif "rnaSequence" in item:
                block = item["rnaSequence"]
                ids = [next(chain_ids) for _ in range(int(block.get("count", 1)))]
                sequences.append(
                    self.RNAInput(
                        id=ids if len(ids) > 1 else ids[0],
                        sequence=block["sequence"],
                    )
                )
            elif "ligand" in item:
                block = item["ligand"]
                ids = [next(chain_ids) for _ in range(int(block.get("count", 1)))]
                ligand_value = block.get("ligand") or block.get("smiles")
                ccd = block.get("ccd") or block.get("ccdCodes")
                if isinstance(ligand_value, str) and ligand_value.startswith("FILE_"):
                    raise ValueError(
                        "ESMFold2 backend does not support FILE_ ligand inputs yet."
                    )
                if ccd is not None:
                    sequences.append(
                        self.LigandInput(
                            id=ids if len(ids) > 1 else ids[0],
                            ccd=_as_list(ccd),
                        )
                    )
                else:
                    sequences.append(
                        self.LigandInput(
                            id=ids if len(ids) > 1 else ids[0],
                            smiles=ligand_value,
                        )
                    )
        if job.get("covalent_bonds"):
            warnings.warn(
                "ESMFold2 covalent bonds require atom indices; template covalent_bonds "
                "are ignored in this ESMFold2-only runner."
            )
        return self.StructurePredictionInput(sequences=sequences)

    def _load_initial_coords(
        self,
        input_atom_array_path: str,
        features: dict[str, Any],
        chain_infos: list[Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        if not input_atom_array_path:
            return None
        if not os.path.exists(input_atom_array_path):
            return None

        atom_mask = features["atom_attention_mask"][0].detach().cpu().bool()
        n_atoms = int(atom_mask.shape[0])

        pdb_path = (
            input_atom_array_path
            if input_atom_array_path.lower().endswith(".pdb")
            else os.path.splitext(input_atom_array_path)[0] + ".pdb"
        )
        if os.path.exists(pdb_path):
            mapped = self._map_pdb_coords_to_features(pdb_path, features, chain_infos)
            if mapped is not None:
                coords, fixed_atom_mask = mapped
                return self._center_coords(coords, atom_mask), fixed_atom_mask
            return None

        try:
            coords = torch.load(input_atom_array_path, map_location="cpu")
            if isinstance(coords, np.ndarray):
                coords = torch.from_numpy(coords)
            coords = coords.float()
            if coords.dim() == 3 and coords.shape[0] == 1:
                coords = coords[0]
            if coords.shape == (n_atoms, 3):
                return self._center_coords(coords, atom_mask), None
        except Exception as exc:
            print(f"Unable to load ESMFold2 initial coordinate tensor: {exc}")

        return None

    def _center_coords(
        self, coords: torch.Tensor, atom_mask: torch.Tensor
    ) -> torch.Tensor:
        coords = coords.float().clone()
        valid = atom_mask.to(dtype=torch.bool)
        centroid = coords[valid].mean(dim=0, keepdim=True)
        coords = coords - centroid
        coords[~valid] = 0.0
        return coords.unsqueeze(0)

    def _map_pdb_coords_to_features(
        self,
        pdb_path: str,
        features: dict[str, Any],
        chain_infos: list[Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        try:
            from Bio.PDB import PDBParser
        except Exception:
            return None

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("init", pdb_path)
        residues_by_chain: dict[str, list[dict[str, np.ndarray]]] = {}
        atoms_by_chain: dict[str, list[tuple[str, str, np.ndarray]]] = {}
        for model in structure:
            for chain in model:
                chain_residues = []
                chain_atoms = []
                for residue in chain:
                    atom_lookup = {}
                    for atom in residue:
                        atom_name = atom.name.strip()
                        atom_lookup[atom_name] = atom.coord
                        if atom.element.upper() != "H":
                            chain_atoms.append((atom_name, atom.element.upper(), atom.coord))
                    chain_residues.append(atom_lookup)
                residues_by_chain[chain.id] = chain_residues
                atoms_by_chain[chain.id] = chain_atoms
            break

        atom_names = features["ref_atom_name_chars"][0].detach().cpu()
        atom_elements = features["ref_element"][0].detach().cpu()
        n_atoms = int(features["atom_attention_mask"][0].shape[0])
        coords = features["ref_pos"][0].detach().cpu().float().clone()
        found = torch.zeros(n_atoms, dtype=torch.bool)
        chain_match_stats = []

        for chain in chain_infos:
            chain_id = chain.chain_id
            pdb_residues = residues_by_chain.get(chain_id, [])
            is_polymer = int(chain.mol_type) in (0, 1, 2)
            chain_required = 0
            chain_matched = 0
            chain_atom_indices = []
            for token in chain.tokens:
                atom_lookup = (
                    pdb_residues[token.residue_index]
                    if token.residue_index < len(pdb_residues)
                    else {}
                )
                for atom_index in range(token.atom_start, token.atom_start + token.atom_count):
                    chain_required += 1
                    chain_atom_indices.append(atom_index)
                    atom_name = _decode_atom_name(atom_names[atom_index])
                    coord = atom_lookup.get(atom_name)
                    if coord is None and not is_polymer:
                        coord = self._match_nonpolymer_atom(
                            atom_lookup,
                            atom_name,
                            _element_symbol(atom_elements[atom_index]),
                        )
                    if coord is None:
                        continue
                    coords[atom_index] = torch.tensor(coord, dtype=torch.float32)
                    found[atom_index] = True
                    chain_matched += 1

            if not is_polymer and chain_required and chain_matched / chain_required < 0.95:
                order_mapped = self._map_nonpolymer_by_atom_order(
                    atoms_by_chain.get(chain_id, []),
                    chain_atom_indices,
                    atom_elements,
                    coords,
                    found,
                )
                if order_mapped:
                    chain_matched = chain_required

            if chain_required and chain_matched / chain_required < 0.95:
                for token in chain.tokens:
                    for atom_index in range(token.atom_start, token.atom_start + token.atom_count):
                        coords[atom_index] = features["ref_pos"][0, atom_index].detach().cpu().float()
                        found[atom_index] = False
                if not is_polymer:
                    print(
                        f"ESMFold2 coordinate mapping matched {chain_matched}/{chain_required} atoms "
                        f"for non-polymer chain {chain_id}; using ESMFold2 reference coordinates for that chain."
                    )
                    chain_match_stats.append((chain_id, chain_matched, chain_required))
                    continue
                print(
                    f"ESMFold2 coordinate mapping matched {chain_matched}/{chain_required} atoms "
                    f"for polymer chain {chain_id}; not using current coordinates for refinement."
                )
                return None
            chain_match_stats.append((chain_id, chain_matched, chain_required))

        polymer_found = False
        polymer_matched = False
        for chain, (_chain_id, matched, required) in zip(chain_infos, chain_match_stats):
            if int(chain.mol_type) in (0, 1, 2):
                polymer_found = True
                if required and matched / required >= 0.95:
                    polymer_matched = True

        if not polymer_found or not polymer_matched:
            print(
                "ESMFold2 coordinate mapping found no mapped polymer chain; "
                "not using current coordinates for refinement."
            )
            return None

        for chain_id, matched, required in chain_match_stats:
            if required == 0:
                print(
                    f"ESMFold2 coordinate mapping matched {matched}/{required} atoms "
                    f"for chain {chain_id}; not using current coordinates for refinement."
                )
                return None

        atom_mask = features["atom_attention_mask"][0].detach().cpu().bool()
        mapped_chain_atom_mask = torch.zeros(n_atoms, dtype=torch.bool)
        for chain, (_chain_id, matched, required) in zip(chain_infos, chain_match_stats):
            if int(chain.mol_type) in (0, 1, 2) and required and matched / required >= 0.95:
                for token in chain.tokens:
                    for atom_index in range(token.atom_start, token.atom_start + token.atom_count):
                        mapped_chain_atom_mask[atom_index] = True

        required = int((atom_mask & mapped_chain_atom_mask).sum().item())
        matched = int((found & atom_mask & mapped_chain_atom_mask).sum().item())
        if required == 0 or matched / required < 0.95:
            print(
                f"ESMFold2 coordinate mapping matched {matched}/{required} atoms; "
                "not using current coordinates for refinement."
            )
            return None
        return coords, None

    def _match_nonpolymer_atom(
        self,
        atom_lookup: dict[str, np.ndarray],
        atom_name: str,
        element: str,
    ) -> np.ndarray | None:
        if not atom_lookup:
            return None
        preferred = atom_name.upper()
        if preferred in atom_lookup:
            return atom_lookup[preferred]
        for pdb_atom_name, coord in atom_lookup.items():
            if pdb_atom_name.upper() == preferred:
                return coord
        if not element:
            return None
        prefix_matches = [
            coord
            for pdb_atom_name, coord in atom_lookup.items()
            if pdb_atom_name.upper().startswith(element)
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        return None

    def _map_nonpolymer_by_atom_order(
        self,
        pdb_atoms: list[tuple[str, str, np.ndarray]],
        atom_indices: list[int],
        atom_elements: torch.Tensor,
        coords: torch.Tensor,
        found: torch.Tensor,
    ) -> bool:
        if len(pdb_atoms) != len(atom_indices):
            return False
        for atom_index, (_name, pdb_element, _coord) in zip(atom_indices, pdb_atoms):
            feature_element = _element_symbol(atom_elements[atom_index]).upper()
            if feature_element and pdb_element and feature_element != pdb_element:
                return False
        for atom_index, (_name, _pdb_element, coord) in zip(atom_indices, pdb_atoms):
            coords[atom_index] = torch.tensor(coord, dtype=torch.float32)
            found[atom_index] = True
        return True

    def _format_halludesign_result(
        self,
        result: Any,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        token_mask = features["token_attention_mask"][0].detach().cpu().bool().numpy()
        asym_raw = features["asym_id"][0].detach().cpu().numpy()[token_mask]
        unique_asym = list(dict.fromkeys(asym_raw.tolist()))
        asym_map = {value: idx for idx, value in enumerate(unique_asym)}
        token_asym_id = np.array([asym_map[value] for value in asym_raw], dtype=np.int64)
        n_chains = max(1, len(unique_asym))
        n_tokens = len(token_asym_id)

        ptm = float(result.ptm) if result.ptm is not None else float("nan")
        iptm = float(result.iptm) if result.iptm is not None else ptm
        if np.isnan(iptm):
            iptm = float(result.plddt.mean().item()) if result.plddt is not None else 0.0
        ranking_score = 0.8 * iptm + 0.2 * (ptm if not np.isnan(ptm) else iptm)

        if result.pair_chains_iptm is not None:
            chain_pair_iptm = result.pair_chains_iptm.float()
            if chain_pair_iptm.shape != (n_chains, n_chains):
                chain_pair_iptm = torch.full((n_chains, n_chains), iptm)
        else:
            chain_pair_iptm = torch.full((n_chains, n_chains), iptm)
        chain_iptm = chain_pair_iptm.mean(dim=1)
        chain_ptm = torch.full((n_chains,), ptm if not np.isnan(ptm) else iptm)

        if result.pae is not None:
            pae = result.pae.detach().cpu().float().numpy()
            if pae.shape[0] == len(token_mask):
                pae = pae[np.ix_(token_mask, token_mask)]
        else:
            pae = np.zeros((n_tokens, n_tokens), dtype=np.float32)
        pde = np.zeros_like(pae, dtype=np.float32)

        return {
            "model_name": "ESMFold2",
            "summary_confidence": [
                {
                    "ranking_score": float(ranking_score),
                    "chain_iptm": chain_iptm,
                    "chain_ptm": chain_ptm,
                    "chain_pair_iptm": chain_pair_iptm,
                }
            ],
            "full_data": [
                {
                    "token_pair_pae": pae,
                    "token_pair_pde": pde,
                    "token_asym_id": token_asym_id,
                }
            ],
        }

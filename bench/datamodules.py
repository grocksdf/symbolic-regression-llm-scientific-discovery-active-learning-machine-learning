from typing import Optional, Any

import os
from pathlib import Path

import numpy as np
import h5py
import datasets
from huggingface_hub import snapshot_download

from .dataclasses import Equation, Problem

REPO_ID = "nnheui/llm-srbench"
LOCAL_DATA_DIR_ENV = "LLM_SRBENCH_DATA_DIR"


def _dataset_root(root: Optional[str] = None) -> Path:
    env_root = os.getenv(LOCAL_DATA_DIR_ENV, "").strip()
    if root is not None and str(root).strip():
        return Path(root)
    if env_root:
        return Path(env_root)
    cwd_candidate = Path.cwd() / "llm-srbench-data"
    if cwd_candidate.exists():
        return cwd_candidate
    sibling_candidate = Path.cwd().parent / "llm-srbench-data"
    if sibling_candidate.exists():
        return sibling_candidate
    file_sibling_candidate = Path(__file__).resolve().parents[2] / "llm-srbench-data"
    if file_sibling_candidate.exists():
        return file_sibling_candidate
    return Path("llm-srbench-data")


def _download(repo_id: str):
    raise RuntimeError("Online dataset download is disabled in this offline-first experiment setup")


def _resolve_local_dataset_root(root: Path, dataset_identifier: str) -> Path:
    root = Path(root)
    candidates = [
        root,
        root / "data",
        root.parent / "data",
        root.parent.parent / "llm-srbench-data",
        root.parent.parent / "llm-srbench-data" / "data",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if _resolve_hdf5_path(candidate).exists():
            return candidate
        if dataset_identifier == "lsrtransform" and (_dataset_parquet_path(candidate, dataset_identifier).exists() or (candidate / "README.md").exists()):
            return candidate
        if dataset_identifier != "lsrtransform" and _dataset_parquet_path(candidate, dataset_identifier).exists():
            return candidate
    return root


def _dataset_parquet_path(root: Path, dataset_identifier: str) -> Path:
    name_map = {
        "phys_osc": "data/lsr_synth_phys_osc-00000-of-00001.parquet",
        "bio_pop_growth": "data/lsr_synth_bio_pop_growth-00000-of-00001.parquet",
        "chem_react": "data/lsr_synth_chem_react-00000-of-00001.parquet",
        "matsci": "data/lsr_synth_matsci-00000-of-00001.parquet",
        "lsrtransform": "data/lsr_transform-00000-of-00001.parquet",
    }
    if dataset_identifier not in name_map:
        raise ValueError(f"Unknown dataset identifier: {dataset_identifier}")
    return Path(root) / name_map[dataset_identifier]


def _load_dataset_from_parquet(root: Path, dataset_identifier: str):
    parquet_path = _dataset_parquet_path(root, dataset_identifier)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {parquet_path}")
    return datasets.load_dataset("parquet", data_files=str(parquet_path), split="train")


def _load_local_hf_dataset(root: Path, split_name: str):
    root = Path(root)
    if not (root / "README.md").exists():
        raise FileNotFoundError(f"Local dataset README not found at {root / 'README.md'}")
    return datasets.load_dataset(str(root), split=split_name)


def _resolve_data_root(root: Optional[str] = None) -> Path:
    base = _dataset_root(root)
    if base.name == "data":
        return base
    if (base / "data").exists() and not (base / "lsr_bench_data.hdf5").exists():
        return base / "data"
    if (base / "llm-srbench-data").exists():
        return base / "llm-srbench-data"
    return base


def _load_problem_rows(root: Path, dataset_identifier: str, split_name: str):
    root = _resolve_data_root(str(root))
    if (root / "README.md").exists():
        try:
            ds = datasets.load_dataset(str(root), split=split_name)
            if ds is not None:
                return ds
        except Exception:
            pass
    if root.exists():
        parquet_path = _dataset_parquet_path(root, dataset_identifier)
        if parquet_path.exists():
            return _load_dataset_from_parquet(root, dataset_identifier)
    return None


def _load_from_hdf5(root: Path, dataset_identifier: str):
    h5_path = _resolve_hdf5_path(root)
    if not h5_path.exists():
        return None
    problems = []
    prefix = "lsr_transform" if dataset_identifier == "lsrtransform" else f"lsr_synth/{dataset_identifier}"
    with h5py.File(h5_path, "r") as sample_file:
        if prefix not in sample_file:
            return None
        for group_name, group in sample_file[prefix].items():
            if not isinstance(group, h5py.Group):
                continue
            samples = {k: v[...].astype(np.float64) for k, v in group.items()}
            problems.append((group_name, samples))
    return problems


def _resolve_hdf5_path(root: Path) -> Path:
    root = Path(root)
    candidates = [root / "lsr_bench_data.hdf5", root / "data" / "lsr_bench_data.hdf5"]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


class TransformedFeynmanDataModule:
    def __init__(self, root=None):
        self._dataset_dir = _resolve_data_root(root)
        self._dataset_identifier = "lsrtransform"

    def setup(self):
        self._dataset_dir = _resolve_local_dataset_root(self._dataset_dir, self._dataset_identifier)
        ds = _load_problem_rows(self._dataset_dir, self._dataset_identifier, "lsr_transform")
        if ds is None:
            ds = datasets.load_dataset(REPO_ID, split="lsr_transform")
            self._dataset_dir = Path(_download(repo_id=REPO_ID))

        sample_rows = _load_from_hdf5(self._dataset_dir, self._dataset_identifier)
        self.problems = []
        if sample_rows is None:
            raise FileNotFoundError(f"Missing required lsr_bench_data.hdf5 under {self._dataset_dir}")
        sample_map = {name: samples for name, samples in sample_rows}
        for e in ds:
            name = e["name"]
            if name not in sample_map:
                continue
            self.problems.append(
                Problem(
                    dataset_identifier=self._dataset_identifier,
                    equation_idx=name,
                    gt_equation=Equation(
                        symbols=e["symbols"],
                        symbol_descs=e["symbol_descs"],
                        symbol_properties=e["symbol_properties"],
                        expression=e["expression"],
                    ),
                    samples=sample_map[name],
                )
            )
        self.name2id = {p.equation_idx: i for i, p in enumerate(self.problems)}

    @property
    def name(self):
        return "LSR_Transform"


class SynProblem(Problem):
    @property
    def train_samples(self):
        return self.samples["train_data"]

    @property
    def test_samples(self):
        return self.samples["id_test_data"]

    @property
    def ood_test_samples(self):
        return self.samples["ood_test_data"]


class BaseSynthDataModule:
    def __init__(self, dataset_identifier, short_dataset_identifier, root, default_symbols=None, default_symbol_descs=None):
        self._dataset_dir = _resolve_data_root(root)
        self._dataset_identifier = dataset_identifier
        self._short_dataset_identifier = short_dataset_identifier
        self._default_symbols = default_symbols
        self._default_symbol_descs = default_symbol_descs

    def setup(self):
        self._dataset_dir = _resolve_local_dataset_root(self._dataset_dir, self._dataset_identifier)
        ds = _load_problem_rows(self._dataset_dir, self._dataset_identifier, f"lsr_synth_{self._dataset_identifier}")
        if ds is None:
            parquet_path = _dataset_parquet_path(self._dataset_dir, self._dataset_identifier)
            h5_path = _resolve_hdf5_path(self._dataset_dir)
            raise FileNotFoundError(
                f"Could not load local dataset rows for {self._dataset_identifier} under {self._dataset_dir}. "
                f"Expected parquet metadata at {parquet_path}. Expected samples at {h5_path}. "
                "Set --ds_root_folder / LLM_SRBENCH_DATA_DIR to a complete local llm-srbench-data checkout."
            )

        sample_rows = _load_from_hdf5(self._dataset_dir, self._dataset_identifier)
        self.problems = []
        if sample_rows is None:
            raise FileNotFoundError(f"Missing required lsr_bench_data.hdf5 under {self._dataset_dir}")
        sample_map = {name: samples for name, samples in sample_rows}
        for e in ds:
            name = e["name"]
            if name not in sample_map:
                continue
            self.problems.append(
                Problem(
                    dataset_identifier=self._dataset_identifier,
                    equation_idx=name,
                    gt_equation=Equation(
                        symbols=e["symbols"],
                        symbol_descs=e["symbol_descs"],
                        symbol_properties=e["symbol_properties"],
                        expression=e["expression"],
                    ),
                    samples=sample_map[name],
                )
            )
        self.name2id = {p.equation_idx: i for i, p in enumerate(self.problems)}

    @property
    def name(self):
        return self._dataset_identifier


class MatSciDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__("matsci", "MatSci", root)


class ChemReactKineticsDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__("chem_react", "CRK", root,
                         default_symbols=["dA_dt", "t", "A"],
                         default_symbol_descs=["Rate of change of concentration in chemistry reaction kinetics", "Time", "Concentration at time t"])


class BioPopGrowthDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__("bio_pop_growth", "BPG", root,
                         default_symbols=["dP_dt", "t", "P"],
                         default_symbol_descs=["Population growth rate", "Time", "Population at time t"])


class PhysOscilDataModule(BaseSynthDataModule):
    def __init__(self, root):
        super().__init__("phys_osc", "PO", root,
                         default_symbols=["dv_dt", "x", "t", "v"],
                         default_symbol_descs=["Acceleration in Nonl-linear Harmonic Oscillator", "Position at time t", "Time", "Velocity at time t"])


def get_datamodule(name, root_folder):
    base_root = _resolve_data_root(root_folder)
    if name == "bio_pop_growth":
        return BioPopGrowthDataModule(base_root)
    elif name == "chem_react":
        return ChemReactKineticsDataModule(base_root)
    elif name == "matsci":
        return MatSciDataModule(base_root)
    elif name == "phys_osc":
        return PhysOscilDataModule(base_root)
    elif name == "lsrtransform":
        return TransformedFeynmanDataModule(base_root)
    else:
        raise ValueError(f"Unknown datamodule name: {name}")

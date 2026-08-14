from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    manuscript_title: str
    target: str
    primary_metric: str
    random_seed: int
    repeats: int
    test_size: float
    min_composition_test_n: int
    split_types: list[str]
    main_models: list[str]
    predefined_main_model: str
    primary_comparisons: list[str]
    conformal_alpha: float
    few_shot_repeats: int
    few_shot_levels: list[int]
    minimum_remaining_test_n_for_few_shot: int


def load_config(path: str | Path = "configs/default_config.json") -> PipelineConfig:
    data = json.loads(Path(path).read_text())
    return PipelineConfig(**data)


def write_run_manifest(config: PipelineConfig, output_path: str | Path, data_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.__dict__ | {"data_path": str(data_path)}
    output_path.write_text(json.dumps(payload, indent=2))

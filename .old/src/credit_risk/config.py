from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class DataCfg:
    path: str
    target: str
    test_size: float
    random_state: int


@dataclass
class ModelCfg:
    type: str
    class_weight: Optional[str] = None
    n_jobs: int = -1


@dataclass
class TuningCfg:
    cv: int
    param_grid: dict


@dataclass
class MlflowCfg:
    experiment: str
    run_name: str
    tracking_uri: Optional[str] = None
    register_model: Optional[str] = None
    register_alias: Optional[str] = None


@dataclass
class Cfg:
    data: DataCfg
    model: ModelCfg
    tuning: TuningCfg
    mlflow: MlflowCfg


def load_cfg(path: str = "params.yaml") -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return Cfg(
        data=DataCfg(**y["data"]),
        model=ModelCfg(**y["model"]),
        tuning=TuningCfg(**y["tuning"]),
        mlflow=MlflowCfg(**y["mlflow"]),
    )

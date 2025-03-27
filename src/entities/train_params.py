from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass()
class LogRegParams:
    model_type: str = field(default="LogisticRegression")
    penalty: str = field(default="l2")
    tol: float = field(default=1e-4)
    random_state: int = field(default=21)


@dataclass()
class RandomForestParams:
    model_type: str = field(default="RandomForestClassifier")
    random_state: int = field(default=21)
    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    })


@dataclass()
class MLPParams:
    model_type: str = field(default="MLPClassifier")
    hidden_layer_sizes: str = field(default="128")
    max_iter: int = field(default=300)
    random_state: int = field(default=21)


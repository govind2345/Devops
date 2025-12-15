from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataConfig:
    id_cols: List[str]
    numeric_cols: List[str]
    label_col: Optional[str]


def load_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")
    return pd.read_csv(csv_path)


def build_preprocessor(cfg: DataConfig) -> ColumnTransformer:
    transformers = []
    
    transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), cfg.id_cols))
    transformers.append(("numeric", StandardScaler(), cfg.numeric_cols))
    
    return ColumnTransformer(transformers)


def split_features_labels(df: pd.DataFrame, cfg: DataConfig):
    X = df[cfg.id_cols + cfg.numeric_cols]
    y = df[cfg.label_col] if cfg.label_col in df.columns else None
    return X, y

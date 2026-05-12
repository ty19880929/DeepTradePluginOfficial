"""LightGBM 模型 / 训练快照的本地存储路径解析。

设计文档 §3.4：模型文件落在 ``<paths.db_path().parent>/volume_anomaly/`` 下，
与 DB 文件同根目录，便于跨平台路径行为一致 + 卸载时一并清理。

未来框架若暴露 ``paths.plugin_data_dir(plugin_id)``，本模块可切到该 API；
当前先用同目录树兜底。
"""

from __future__ import annotations

from pathlib import Path

from deeptrade.core import paths


def plugin_data_dir() -> Path:
    """``~/.deeptrade/volume_anomaly/`` —— 插件持久化数据根目录。"""
    return paths.db_path().parent / "volume_anomaly"


def models_dir() -> Path:
    """已训练 LightGBM 模型文件目录。"""
    return plugin_data_dir() / "models"


def datasets_dir() -> Path:
    """训练矩阵 parquet 快照目录（每个 model_id 一份，便于复现）。"""
    return plugin_data_dir() / "datasets"


def checkpoints_dir() -> Path:
    """Phase-1 续训目录根。各 fingerprint 一个子目录，包含 days/<YYYYMMDD>.parquet。"""
    return plugin_data_dir() / "checkpoints"


def latest_pointer() -> Path:
    """单行文本指针，存当前 active 模型在 ``models_dir`` 内的相对路径。"""
    return models_dir() / "latest.txt"


def ensure_layout() -> None:
    """Idempotent: create models/ + datasets/ + checkpoints/ if missing."""
    models_dir().mkdir(parents=True, exist_ok=True)
    datasets_dir().mkdir(parents=True, exist_ok=True)
    checkpoints_dir().mkdir(parents=True, exist_ok=True)


def model_file_name(model_id: str) -> str:
    """约定文件名：``lgb_model_<model_id>.txt``."""
    return f"lgb_model_{model_id}.txt"


def meta_file_name(model_id: str) -> str:
    """模型元信息 JSON 文件名（与 model file 同前缀）。"""
    return f"lgb_model_{model_id}.meta.json"


def dataset_file_name(model_id: str) -> str:
    """训练矩阵 parquet 文件名。"""
    return f"lgb_dataset_{model_id}.parquet"

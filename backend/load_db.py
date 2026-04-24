from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetLoadConfig:
    database_path: str
    split: str = "train"


class DatasetLoader:
    """Loads TFDS datasets from a prepared dataset directory."""

    def __init__(self, config: DatasetLoadConfig) -> None:
        self._config = config

    def load(self) -> Any:
        import tensorflow_datasets as tfds

        version_dir = self._resolve_version_directory(Path(self._config.database_path))
        builder = tfds.builder_from_directory(str(version_dir))
        return builder.as_dataset(split=self._config.split)

    def _resolve_version_directory(self, db_path: Path) -> Path:
        if self._is_version_directory(db_path):
            return db_path

        if not db_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {db_path}")

        version_dirs = [entry for entry in db_path.iterdir() if self._is_version_directory(entry)]
        if not version_dirs:
            raise FileNotFoundError(
                "No TFDS version directory found. Expected a folder containing dataset_info.json."
            )
        return sorted(version_dirs)[-1]

    @staticmethod
    def _is_version_directory(path: Path) -> bool:
        return path.is_dir() and (path / "dataset_info.json").exists()

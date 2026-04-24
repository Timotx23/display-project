from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

from backend.load_db import DatasetLoadConfig, DatasetLoader


@dataclass(frozen=True)
class VideoConfig:
    episode: int = 0
    preferred_frame_key: Optional[str] = None


@dataclass(frozen=True)
class MultiCameraFrame:
    step_index: int
    main_frame: np.ndarray
    wrist_frame: Optional[np.ndarray]
    joint_angles: Optional[list[float]]


class VideoReader:
    """Video-only logic: extracts frames from RLDS episode steps."""

    def __init__(self, dataset: Any, config: VideoConfig) -> None:
        self._dataset = dataset
        self._config = config

    def read_episode_frames(self) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for frame in self.iter_episode_frames():
            frames.append(frame)
        return frames

    def get_episode_prompt(self) -> Optional[str]:
        """Return the episode-level user prompt/instruction if available."""
        import tensorflow_datasets as tfds

        if self._config.episode < 0:
            raise IndexError("Episode index must be >= 0.")

        for episode_index, episode in enumerate(tfds.as_numpy(self._dataset)):
            if episode_index != self._config.episode:
                continue
            return self._find_prompt_text(episode)

        raise IndexError(f"Episode index {self._config.episode} out of range.")

    def iter_episode_frames(self) -> Iterator[np.ndarray]:
        """Yield frames from one selected episode (old_vid.py behavior)."""
        import tensorflow_datasets as tfds

        if self._config.episode < 0:
            raise IndexError("Episode index must be >= 0.")

        for episode_index, episode in enumerate(tfds.as_numpy(self._dataset)):
            if episode_index != self._config.episode:
                continue
            steps = episode.get("steps")
            if steps is None:
                raise KeyError("Episode does not contain 'steps'.")

            for step in steps:
                frame = self._extract_frame(step)
                if frame is not None:
                    yield frame
            return

        raise IndexError(f"Episode index {self._config.episode} out of range.")

    def iter_multicamera_episode_frames(self) -> Iterator[MultiCameraFrame]:
        """Yield synchronized main/wrist frames for the selected episode."""
        import tensorflow_datasets as tfds

        if self._config.episode < 0:
            raise IndexError("Episode index must be >= 0.")

        for episode_index, episode in enumerate(tfds.as_numpy(self._dataset)):
            if episode_index != self._config.episode:
                continue
            steps = episode.get("steps")
            if steps is None:
                raise KeyError("Episode does not contain 'steps'.")

            for step_index, step in enumerate(steps):
                candidates = self._find_candidate_frames(step)
                main = self._pick_frame(candidates)
                if main is None:
                    continue

                wrist = self._pick_wrist_frame(candidates)
                joint_angles = self._extract_joint_angles(step)
                yield MultiCameraFrame(
                    step_index=step_index,
                    main_frame=self._to_uint8(main),
                    wrist_frame=self._to_uint8(wrist) if wrist is not None else None,
                    joint_angles=joint_angles,
                )
            return

        raise IndexError(f"Episode index {self._config.episode} out of range.")

    def _extract_frame(self, step: dict[str, Any]) -> Optional[np.ndarray]:
        candidates = self._find_candidate_frames(step)
        frame = self._pick_frame(candidates)
        if frame is None:
            return None
        return self._to_uint8(frame)

    def _to_numpy(self, value: Any) -> Any:
        if hasattr(value, "numpy"):
            return value.numpy()
        return value

    def _find_prompt_text(self, episode: dict[str, Any]) -> Optional[str]:
        prompt_key_hints = (
            "language_instruction",
            "instruction",
            "prompt",
            "task",
            "command",
            "goal",
            "text",
        )

        ranked_matches: list[tuple[int, str]] = []

        def visit(node: Any, key_path: str = "") -> None:
            if isinstance(node, dict):
                for key, child in node.items():
                    next_path = f"{key_path}.{key}" if key_path else str(key)
                    visit(child, next_path)
                return
            if isinstance(node, (list, tuple)):
                for idx, child in enumerate(node):
                    next_path = f"{key_path}[{idx}]"
                    visit(child, next_path)
                return

            value = self._to_numpy(node)
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8")
                except Exception:
                    return
            if not isinstance(value, str):
                return

            normalized = " ".join(value.strip().split())
            if not normalized:
                return

            key_lower = key_path.lower()
            if "step" in key_lower and "language_instruction" not in key_lower:
                return

            for rank, hint in enumerate(prompt_key_hints):
                if hint in key_lower:
                    ranked_matches.append((rank, normalized))
                    break

        visit(episode)
        if not ranked_matches:
            return None
        ranked_matches.sort(key=lambda item: item[0])
        return ranked_matches[0][1]

    def _find_candidate_frames(self, value: Any) -> list[tuple[str, np.ndarray]]:
        candidates: list[tuple[str, np.ndarray]] = []

        def visit(node: Any, prefix: str = "") -> None:
            if isinstance(node, dict):
                for key, child in node.items():
                    next_prefix = f"{prefix}.{key}" if prefix else str(key)
                    visit(child, next_prefix)
                return

            array = self._to_numpy(node)
            if isinstance(array, np.ndarray):
                # Match old_vid.py candidate constraints.
                if (
                    array.ndim == 3
                    and array.shape[0] > 8
                    and array.shape[1] > 8
                    and array.shape[2] in (1, 3, 4)
                ):
                    candidates.append((prefix, array))
                elif array.ndim == 4 and array.shape[-1] in (1, 3, 4):
                    candidates.append((prefix, array[0]))

        visit(value)
        return candidates

    def _pick_frame(self, candidates: list[tuple[str, np.ndarray]]) -> Optional[np.ndarray]:
        if not candidates:
            return None

        preferred_key = self._config.preferred_frame_key
        if preferred_key:
            for key_path, frame in candidates:
                if key_path == preferred_key or key_path.endswith(f".{preferred_key}"):
                    return frame

        for suffix in (
            "observation.image",
            "observation.rgb",
            "observation.camera",
            "observation.front_camera",
            "image",
            "rgb",
        ):
            for key_path, frame in candidates:
                if key_path == suffix or key_path.endswith(f".{suffix}"):
                    return frame

        return candidates[0][1]

    def _to_uint8(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype == np.uint8:
            return frame
        if np.issubdtype(frame.dtype, np.floating):
            clipped = np.clip(frame, 0.0, 1.0)
            return (clipped * 255.0).astype(np.uint8)
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _pick_wrist_frame(self, candidates: list[tuple[str, np.ndarray]]) -> Optional[np.ndarray]:
        if not candidates:
            return None

        wrist_suffixes = (
            "observation.wrist",
            "observation.wrist_camera",
            "observation.camera_wrist",
            "observation.hand_camera",
            "wrist",
            "wrist_camera",
            "camera_wrist",
            "hand_camera",
        )
        for suffix in wrist_suffixes:
            for key_path, frame in candidates:
                if key_path == suffix or key_path.endswith(f".{suffix}") or "wrist" in key_path.lower():
                    return frame
        return None

    def _extract_joint_angles(self, step: dict[str, Any]) -> Optional[list[float]]:
        key_hints = ("joint", "motor", "qpos", "angle", "position", "action")
        scored_candidates: list[tuple[int, list[float]]] = []

        def visit(node: Any, key_path: str = "") -> None:
            if isinstance(node, dict):
                for key, child in node.items():
                    next_path = f"{key_path}.{key}" if key_path else str(key)
                    visit(child, next_path)
                return
            if isinstance(node, (list, tuple)):
                for idx, child in enumerate(node):
                    next_path = f"{key_path}[{idx}]"
                    visit(child, next_path)
                return

            arr = self._to_numpy(node)
            if not isinstance(arr, np.ndarray):
                return
            if arr.ndim != 1 or arr.size < 6 or arr.size > 32:
                return
            if np.issubdtype(arr.dtype, np.integer) and arr.max(initial=0) <= 255:
                return
            if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
                values = arr.astype(float).tolist()[:6]
                key_lower = key_path.lower()
                score = 100
                for rank, hint in enumerate(key_hints):
                    if hint in key_lower:
                        score = rank
                        break
                scored_candidates.append((score, values))

        visit(step)
        if not scored_candidates:
            return None
        scored_candidates.sort(key=lambda item: item[0])
        return scored_candidates[0][1]


class VideoService:
    """Coordinates DB loading and video frame extraction."""

    def load_episode_frames(
        self,
        database_path: str,
        episode: int,
        split: str = "train",
        preferred_frame_key: Optional[str] = None,
    ) -> list[np.ndarray]:
        dataset_loader = DatasetLoader(DatasetLoadConfig(database_path=database_path, split=split))
        dataset = dataset_loader.load()
        video_reader = VideoReader(
            dataset=dataset,
            config=VideoConfig(episode=episode, preferred_frame_key=preferred_frame_key),
        )
        return video_reader.read_episode_frames()

    def get_episode_prompt(
        self,
        database_path: str,
        episode: int,
        split: str = "train",
        preferred_frame_key: Optional[str] = None,
    ) -> Optional[str]:
        dataset_loader = DatasetLoader(DatasetLoadConfig(database_path=database_path, split=split))
        dataset = dataset_loader.load()
        video_reader = VideoReader(
            dataset=dataset,
            config=VideoConfig(episode=episode, preferred_frame_key=preferred_frame_key),
        )
        return video_reader.get_episode_prompt()

    def iter_episode_frames(
        self,
        database_path: str,
        episode: int,
        split: str = "train",
        preferred_frame_key: Optional[str] = None,
    ) -> Iterator[np.ndarray]:
        dataset_loader = DatasetLoader(DatasetLoadConfig(database_path=database_path, split=split))
        dataset = dataset_loader.load()
        video_reader = VideoReader(
            dataset=dataset,
            config=VideoConfig(episode=episode, preferred_frame_key=preferred_frame_key),
        )
        return video_reader.iter_episode_frames()

    def iter_multicamera_episode_frames(
        self,
        database_path: str,
        episode: int,
        split: str = "train",
        preferred_frame_key: Optional[str] = None,
    ) -> Iterator[MultiCameraFrame]:
        dataset_loader = DatasetLoader(DatasetLoadConfig(database_path=database_path, split=split))
        dataset = dataset_loader.load()
        video_reader = VideoReader(
            dataset=dataset,
            config=VideoConfig(episode=episode, preferred_frame_key=preferred_frame_key),
        )
        return video_reader.iter_multicamera_episode_frames()

    def get_dataset_joint_statistics(
        self,
        database_path: str,
        split: str = "train",
    ) -> dict[str, dict[str, float]]:
        """
        Compute dataset-level descriptive stats for joint sensors across all episodes/steps.
        Uses online accumulation for memory efficiency.
        """
        import tensorflow_datasets as tfds

        dataset_loader = DatasetLoader(DatasetLoadConfig(database_path=database_path, split=split))
        dataset = dataset_loader.load()
        reader = VideoReader(dataset=dataset, config=VideoConfig())

        joint_count = 6
        counts = np.zeros(joint_count, dtype=np.int64)
        means = np.zeros(joint_count, dtype=np.float64)
        m2 = np.zeros(joint_count, dtype=np.float64)
        mins = np.full(joint_count, np.inf, dtype=np.float64)
        maxs = np.full(joint_count, -np.inf, dtype=np.float64)

        for episode in tfds.as_numpy(dataset):
            steps = episode.get("steps")
            if steps is None:
                continue
            for step in steps:
                joint_angles = reader._extract_joint_angles(step)
                if joint_angles is None or len(joint_angles) < joint_count:
                    continue
                for idx, raw_value in enumerate(joint_angles[:joint_count]):
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(value):
                        continue

                    counts[idx] += 1
                    delta = value - means[idx]
                    means[idx] += delta / counts[idx]
                    delta2 = value - means[idx]
                    m2[idx] += delta * delta2
                    mins[idx] = min(mins[idx], value)
                    maxs[idx] = max(maxs[idx], value)

        stats: dict[str, dict[str, float]] = {}
        for idx in range(joint_count):
            sensor_name = f"Joint {idx + 1}"
            if counts[idx] == 0:
                stats[sensor_name] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
                continue
            variance = m2[idx] / counts[idx]
            stats[sensor_name] = {
                "mean": float(means[idx]),
                "std": float(np.sqrt(variance)),
                "min": float(mins[idx]),
                "max": float(maxs[idx]),
            }
        return stats



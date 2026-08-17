"""Batch TorchXRayVision inference for chest X-ray image files.

This script intentionally does not compute AUROC.  AUROC must be computed later
against the team's frozen split and ground-truth/Image Index mapping.

Examples
--------
    python inference.py --weights densenet121-res224-all image1.png image2.png
    python inference.py --weights resnet50-res512-all image1.png --output resnet_scores.parquet
    python inference.py --weights densenet121-res224-nih \
        --images-file smoke_images.txt --output nih_scores.parquet

The pretrained weights must already be present in TorchXRayVision's local
cache (or in ``--cache-dir``).  This script never downloads weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


SUPPORTED_WEIGHTS = (
    "densenet121-res224-nih",
    "densenet121-res224-all",
    "resnet50-res512-all",
)

MODEL_CONFIGS = {
    "densenet121-res224-nih": {
        "loader": "DenseNet",
        "backbone": "torchxrayvision.DenseNet121",
        "input_resolution": 224,
    },
    "densenet121-res224-all": {
        "loader": "DenseNet",
        "backbone": "torchxrayvision.DenseNet121",
        "input_resolution": 224,
    },
    "resnet50-res512-all": {
        "loader": "ResNet",
        "backbone": "torchxrayvision.ResNet50",
        "input_resolution": 512,
    },
}

# This is the canonical order used by TorchXRayVision for the NIH-trained
# output slots.  The NIH weight file has four additional blank slots after
# these 14 outputs; those slots must never become unnamed parquet columns.
NIH_FINDING_NAMES = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
]

ALL_FINDING_NAMES = NIH_FINDING_NAMES + [
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
]

EXPECTED_MODEL_LABELS = {
    "densenet121-res224-nih": NIH_FINDING_NAMES + ["", "", "", ""],
    "densenet121-res224-all": ALL_FINDING_NAMES,
    "resnet50-res512-all": ALL_FINDING_NAMES,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run TorchXRayVision chest X-ray inference on one or more raster "
            "chest X-ray images and write sigmoid scores to parquet. "
            "Weights are local-only; no downloads are performed."
        ),
        epilog=(
            "Images may be supplied as repeated positional paths, with one path "
            "per line in --images-file, or both. Example: python inference.py "
            "--weights densenet121-res224-nih --images-file images.txt "
            "--expected-labels-file nih_labels.txt --output scores.parquet"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "images",
        nargs="*",
        metavar="IMAGE",
        help="Image path; repeat for multiple images. Use -- before a path beginning with '-'.",
    )
    parser.add_argument(
        "--images-file",
        action="append",
        default=[],
        metavar="PATH",
        help="Text file containing one image path per line; may be repeated.",
    )
    parser.add_argument(
        "--weights",
        choices=SUPPORTED_WEIGHTS,
        default="densenet121-res224-all",
        help="TorchXRayVision pretrained weight name.",
    )
    parser.add_argument(
        "--expected-labels-file",
        metavar="PATH",
        help="Optional file with one expected finding name per line, in required order.",
    )
    parser.add_argument(
        "--expected-label",
        action="append",
        default=None,
        metavar="NAME",
        help="Expected finding name; repeat to provide the exact expected order.",
    )
    parser.add_argument(
        "--output",
        default="predictions.parquet",
        metavar="PATH",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of preprocessed images per model batch.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. 'auto' selects CUDA when available.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader worker processes for image loading/preprocessing.",
    )
    parser.add_argument(
        "--cache-dir",
        metavar="PATH",
        help="TorchXRayVision weight cache containing the already-downloaded file.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optional cap useful for smoke tests; preserves input order.",
    )
    return parser


def _read_image_paths(positional: Sequence[str], list_files: Sequence[str]) -> List[Path]:
    raw_paths = list(positional)
    for list_file in list_files:
        file_path = Path(list_file).expanduser()
        if not file_path.is_file():
            raise FileNotFoundError(f"Image-list file does not exist: {file_path}")
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError(f"Could not read image-list file {file_path}: {exc}") from exc
        raw_paths.extend(
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        )

    if not raw_paths:
        raise ValueError("No images supplied. Provide IMAGE paths and/or --images-file PATH.")

    resolved: List[Path] = []
    seen_indices: Dict[str, Path] = {}
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser().resolve(strict=False)
        if not path.is_file():
            raise FileNotFoundError(f"Image path does not exist or is not a file: {path}")
        image_index = path.name
        previous = seen_indices.get(image_index)
        if previous is not None and previous != path:
            raise ValueError(
                "Image Index collision: two different paths have the same basename "
                f"{image_index!r}: {previous} and {path}."
            )
        if previous is None:
            seen_indices[image_index] = path
            resolved.append(path)
    return resolved


def _read_expected_labels(
    expected_file: Optional[str], expected_values: Optional[Sequence[str]]
) -> Optional[List[str]]:
    if expected_file and expected_values:
        raise ValueError("Use only one of --expected-labels-file and --expected-label.")
    if expected_file:
        path = Path(expected_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Expected-label file does not exist: {path}")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError(f"Could not read expected-label file {path}: {exc}") from exc
        labels = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    elif expected_values:
        labels = [label.strip() for label in expected_values]
    else:
        return None

    if not labels or any(not label for label in labels):
        raise ValueError("Expected labels must contain at least one non-empty name.")
    return labels


def _import_runtime_dependencies() -> Tuple[Any, Any, Any, Any, Any]:
    """Import optional runtime dependencies with actionable error messages."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required. Install the runtime dependencies in the target "
            "environment before running inference."
        ) from exc
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required. Install a CPU or CUDA-compatible PyTorch build "
            "before running inference."
        ) from exc
    try:
        import torchxrayvision as xrv
    except ImportError as exc:
        raise RuntimeError(
            "TorchXRayVision is not installed. Install it in the target runtime "
            "(for example, `pip install torchxrayvision`) and rerun; this script "
            "does not install packages or download weights."
        ) from exc
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires PyArrow. Install `pyarrow` in the target "
            "runtime; no parquet dependency is installed by this script."
        ) from exc
    return np, torch, xrv, pa, pq


def _infer_maxval(np: Any, image: Any) -> float:
    if getattr(image.dtype, "kind", None) in "iu":
        return float(np.iinfo(image.dtype).max)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixel values.")
    maximum = float(np.max(finite))
    minimum = float(np.min(finite))
    if minimum >= 0.0 and maximum <= 1.0:
        return 1.0
    if minimum >= 0.0 and maximum <= 255.0:
        return 255.0
    return max(maximum, 1.0)


def _load_raster_image(path: Path, np: Any) -> Any:
    try:
        from skimage.io import imread

        image = imread(str(path))
    except ImportError:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Image loading requires scikit-image or Pillow. Install one in "
                "the target runtime."
            ) from exc
        try:
            with Image.open(path) as pil_image:
                image = np.asarray(pil_image)
        except Exception as exc:
            raise RuntimeError(f"Could not decode image {path}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not decode image {path}: {exc}") from exc

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] == 1:
        return image[..., 0]
    if image.ndim == 3 and image.shape[-1] >= 3:
        # TorchXRayVision expects a single-channel radiograph. Ignore alpha and
        # use a simple channel mean, matching the documented XRV example.
        return image[..., :3].mean(axis=-1)
    raise ValueError(f"Unsupported image shape {image.shape} for {path}; expected HxW or HxWx3.")


def _center_crop(np: Any, image: Any) -> Any:
    height, width = image.shape[-2:]
    crop_size = min(height, width)
    start_y = height // 2 - crop_size // 2
    start_x = width // 2 - crop_size // 2
    return image[..., start_y : start_y + crop_size, start_x : start_x + crop_size]


class _XRayPreprocessor:
    """Lazy-per-process XRV preprocessing, so DataLoader workers are picklable."""

    def __init__(self, np: Any, torch: Any, xrv: Any, input_resolution: int) -> None:
        self.np = np
        self.torch = torch
        self.xrv = xrv
        self.input_resolution = int(input_resolution)
        if self.input_resolution <= 0:
            raise ValueError(f"input_resolution must be positive, got {input_resolution!r}.")
        self._normalize: Optional[Callable[..., Any]] = None
        self._center_crop_transform: Optional[Callable[[Any], Any]] = None
        self._resizer_transform: Optional[Callable[[Any], Any]] = None
        self.mode = "uninitialized"
        self._initialize()

    def _initialize(self) -> None:
        datasets = getattr(self.xrv, "datasets", None)
        if datasets is None:
            raise RuntimeError("TorchXRayVision has no datasets module; check the installed version.")
        self._normalize = getattr(datasets, "normalize", None)
        if self._normalize is None:
            utils = getattr(self.xrv, "utils", None)
            self._normalize = getattr(utils, "normalize", None) if utils else None
        center_cls = getattr(datasets, "XRayCenterCrop", None)
        resize_cls = getattr(datasets, "XRayResizer", None)
        if callable(center_cls) and callable(resize_cls):
            self._center_crop_transform = center_cls()
            self._resizer_transform = resize_cls(self.input_resolution)
            self.mode = (
                "torchxrayvision.datasets.normalize+XRayCenterCrop+"
                f"XRayResizer({self.input_resolution})"
            )
        else:
            self.mode = (
                "manual_normalize+center_crop+"
                f"torch_interpolate({self.input_resolution})"
            )
            print(
                "Warning: installed TorchXRayVision lacks its documented crop/resize "
                "utilities; using the explicit fallback preprocessing path.",
                file=sys.stderr,
            )
        if self._normalize is None:
            print(
                "Warning: installed TorchXRayVision lacks normalize(); using the "
                "documented [-1024, 1024] scaling formula as fallback.",
                file=sys.stderr,
            )

    def __call__(self, path: Path) -> Any:
        image = _load_raster_image(path, self.np)
        maxval = _infer_maxval(self.np, image)
        image = image.astype(self.np.float32, copy=False)
        if self._normalize is not None:
            try:
                image = self._normalize(image[None, ...], maxval=maxval, reshape=False)
            except TypeError:
                image = self._normalize(image[None, ...], maxval)
        else:
            image = (image[None, ...] / maxval) * 2048.0 - 1024.0

        if self._center_crop_transform is not None and self._resizer_transform is not None:
            image = self._center_crop_transform(image)
            image = self._resizer_transform(image)
        else:
            image = _center_crop(self.np, image)
            tensor = self.torch.from_numpy(self.np.asarray(image, dtype=self.np.float32)).unsqueeze(0)
            tensor = self.torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.input_resolution, self.input_resolution),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            image = tensor.numpy()

        tensor = self.torch.from_numpy(self.np.asarray(image, dtype=self.np.float32))
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        expected_shape = (1, self.input_resolution, self.input_resolution)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Preprocessing produced shape {tuple(tensor.shape)} for {path}, "
                f"expected {expected_shape}."
            )
        if not self.torch.isfinite(tensor).all():
            raise ValueError(f"Preprocessing produced non-finite values for {path}.")
        return tensor


class _ImageDataset:
    def __init__(self, paths: Sequence[Path], input_resolution: int) -> None:
        self.paths = list(paths)
        self.input_resolution = int(input_resolution)
        self._preprocessor: Optional[_XRayPreprocessor] = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[Any, str]:
        path = self.paths[index]
        if self._preprocessor is None:
            # Import inside the worker so the dataset remains pickleable on
            # Windows when --workers is greater than zero.
            import numpy as np
            import torch
            import torchxrayvision as xrv

            self._preprocessor = _XRayPreprocessor(
                np,
                torch,
                xrv,
                input_resolution=self.input_resolution,
            )
        try:
            return self._preprocessor(path), str(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load/preprocess image {path}: {exc}") from exc


def _weight_file_path(xrv: Any, weights: str, cache_dir: Optional[str]) -> Path:
    model_urls = getattr(getattr(xrv, "models", None), "model_urls", None)
    if not isinstance(model_urls, dict) or weights not in model_urls:
        raise RuntimeError(
            "Cannot inspect the TorchXRayVision weight registry in this installed "
            "version. Upgrade TorchXRayVision or provide a version exposing "
            "torchxrayvision.models.model_urls; refusing to risk an implicit download."
        )
    weights_url = model_urls[weights].get("weights_url")
    filename = Path(urlparse(str(weights_url)).path).name if weights_url else ""
    if not filename:
        raise RuntimeError(f"No local-cache filename is registered for weight name {weights!r}.")
    if cache_dir:
        root = Path(cache_dir).expanduser()
    else:
        utils = getattr(xrv, "utils", None)
        get_cache_dir = getattr(utils, "get_cache_dir", None) if utils else None
        if not callable(get_cache_dir):
            raise RuntimeError(
                "TorchXRayVision does not expose its cache directory helper. "
                "Pass --cache-dir pointing to the already-downloaded weights."
            )
        root = Path(get_cache_dir()).expanduser()
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Pretrained weights are not present locally: {path}. Download the "
            "matching TorchXRayVision weight file outside this script, then rerun; "
            "this script never downloads weights."
        )
    return path


def _load_model(xrv: Any, torch: Any, weights: str, cache_dir: Optional[str]) -> Any:
    local_path = _weight_file_path(xrv, weights, cache_dir)
    config = MODEL_CONFIGS[weights]
    loader = getattr(getattr(xrv, "models", None), config["loader"], None)
    if not callable(loader):
        raise RuntimeError(
            f"TorchXRayVision does not expose models.{config['loader']} "
            "in this installation."
        )
    try:
        try:
            model = loader(
                weights=weights,
                cache_dir=str(local_path.parent),
                apply_sigmoid=False,
            )
        except TypeError:
            try:
                model = loader(weights=weights, cache_dir=str(local_path.parent))
            except TypeError:
                model = loader(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            f"TorchXRayVision could not load {weights!r} from local cache {local_path}: {exc}"
        ) from exc
    model.eval()
    return model


def _validated_labels(
    model: Any, weights: str, expected_labels: Optional[Sequence[str]]
) -> Tuple[List[str], List[str], List[int]]:
    raw = getattr(model, "pathologies", None)
    if raw is None:
        raw = getattr(model, "targets", None)
    if raw is None:
        raise RuntimeError("The loaded model exposes neither .pathologies nor .targets; cannot align outputs safely.")
    raw_labels = ["" if label is None else str(label) for label in list(raw)]
    if not raw_labels:
        raise RuntimeError("The loaded model exposes an empty label list; refusing to write unlabeled scores.")

    expected_raw_labels = EXPECTED_MODEL_LABELS[weights]
    if raw_labels != expected_raw_labels:
        raise RuntimeError(
            f"TorchXRayVision label-order safeguard failed for {weights!r}. "
            f"Expected raw model labels: {expected_raw_labels!r}. "
            f"Loaded: {raw_labels!r}. Refusing silent realignment."
        )

    if weights == "densenet121-res224-nih":
        output_indices = [index for index, label in enumerate(raw_labels) if label]
        output_labels = [raw_labels[index] for index in output_indices]
    else:
        output_indices = list(range(len(raw_labels)))
        output_labels = raw_labels

    duplicates = sorted({label for label in output_labels if output_labels.count(label) > 1})
    if duplicates:
        raise RuntimeError(f"Loaded model has duplicate finding names {duplicates}; refusing silent misalignment.")
    if expected_labels is not None and list(expected_labels) != output_labels:
        raise RuntimeError(
            "Expected finding-label order does not exactly match the loaded model.\n"
            f"Expected: {list(expected_labels)!r}\n"
            f"Loaded:   {output_labels!r}\n"
            "Refusing to write scores with a potentially misaligned schema."
        )
    return raw_labels, output_labels, output_indices


def _sigmoid_scores(model: Any, batch: Any, torch: Any) -> Tuple[Any, str]:
    """Return plain sigmoid probabilities, bypassing XRV operating-point remapping."""
    if hasattr(model, "features2") and hasattr(model, "classifier"):
        try:
            logits = model.classifier(model.features2(batch))
            return torch.sigmoid(logits), "sigmoid(model.classifier(model.features2(batch)))"
        except Exception as exc:
            raise RuntimeError("Could not compute DenseNet logits before sigmoid.") from exc

    # TorchXRayVision's ResNet wrapper stores the torchvision ResNet under
    # ``model`` and its public forward method applies operating-point
    # remapping. Use the wrapped logits so DenseNet and ResNet produce the
    # same plain sigmoid-score schema for downstream code.
    if hasattr(model, "model") and hasattr(model, "op_threshs"):
        try:
            logits = model.model(batch)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Model logits contain non-finite values.")
            return torch.sigmoid(logits), "sigmoid(model.model(batch))"
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Could not compute ResNet logits before sigmoid.") from exc

    output = model(batch)
    if not torch.isfinite(output).all():
        raise RuntimeError("Model output contains non-finite values.")
    if bool(((output >= 0.0) & (output <= 1.0)).all()):
        return output, "model_output_already_probability"
    return torch.sigmoid(output), "sigmoid(model_output)"


def _preprocessing_description(xrv: Any, input_resolution: int) -> str:
    datasets = getattr(xrv, "datasets", None)
    if datasets is not None:
        normalize = getattr(datasets, "normalize", None)
        center = getattr(datasets, "XRayCenterCrop", None)
        resizer = getattr(datasets, "XRayResizer", None)
        if normalize is not None and callable(center) and callable(resizer):
            return (
                "torchxrayvision.datasets.normalize+XRayCenterCrop+"
                f"XRayResizer({input_resolution})"
            )
    return f"manual_normalize+center_crop+torch_interpolate({input_resolution})"


def _write_parquet(
    output_path: Path,
    image_paths: Sequence[str],
    scores: Any,
    output_labels: Sequence[str],
    metadata: Dict[str, Any],
    pa: Any,
    pq: Any,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_indices = [Path(path).name for path in image_paths]
    columns: Dict[str, Any] = {
        "Image Index": image_indices,
        "image_path": list(image_paths),
        "backbone": [metadata["backbone"]] * len(image_paths),
        "weights": [metadata["weights"]] * len(image_paths),
    }
    for label_index, label in enumerate(output_labels):
        columns[label] = scores[:, label_index].astype("float32")
    table = pa.table(columns)
    encoded_metadata = {
        str(key).encode("utf-8"): json.dumps(value, sort_keys=True).encode("utf-8")
        for key, value in metadata.items()
    }
    table = table.replace_schema_metadata(encoded_metadata)
    pq.write_table(table, output_path)


def run(args: argparse.Namespace) -> int:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.workers < 0:
        raise ValueError("--workers must be zero or greater.")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive when supplied.")

    paths = _read_image_paths(args.images, args.images_file)
    if args.max_images is not None:
        paths = paths[: args.max_images]
    expected_labels = _read_expected_labels(args.expected_labels_file, args.expected_label)
    np, torch, xrv, pa, pq = _import_runtime_dependencies()
    model_config = MODEL_CONFIGS[args.weights]
    input_resolution = int(model_config["input_resolution"])

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false.")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto"
        else "cpu"
    )
    model = _load_model(xrv, torch, args.weights, args.cache_dir)
    raw_labels, output_labels, output_indices = _validated_labels(model, args.weights, expected_labels)
    model.to(device)

    dataset = _ImageDataset(paths, input_resolution=input_resolution)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    score_rows: List[Any] = []
    output_paths: List[str] = []
    score_transform: Optional[str] = None
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, (batch, batch_paths) in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=device.type == "cuda")
            batch_scores, transform_name = _sigmoid_scores(model, batch, torch)
            if score_transform is None:
                score_transform = transform_name
            elif score_transform != transform_name:
                raise RuntimeError("Score transformation changed within a run; refusing inconsistent output.")
            if batch_scores.shape[1] != len(raw_labels):
                raise RuntimeError(
                    f"Model returned {batch_scores.shape[1]} outputs but exposes {len(raw_labels)} labels."
                )
            selected = batch_scores[:, output_indices].detach().cpu().numpy()
            score_rows.append(selected)
            output_paths.extend(batch_paths)
            print(
                f"Processed {len(output_paths)}/{len(paths)} images (batch {batch_number}).",
                file=sys.stderr,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not score_rows:
        raise RuntimeError("No inference rows were produced.")

    scores = np.concatenate(score_rows, axis=0)
    metadata: Dict[str, Any] = {
        "schema_version": "1",
        "image_index_key": "Path(image_path).name",
        "backbone": model_config["backbone"],
        "weights": args.weights,
        "input_size": f"{input_resolution}x{input_resolution}",
        "preprocessing": _preprocessing_description(xrv, input_resolution),
        "preprocessing_fallback_allowed": True,
        "score_definition": score_transform or "unknown",
        "raw_model_labels": raw_labels,
        "output_labels": output_labels,
        "output_label_indices": output_indices,
        "expected_model_labels": EXPECTED_MODEL_LABELS[args.weights],
        "label_order_validated": True,
        "auroc_validated": False,
        "auroc_note": "AUROC intentionally not computed; team split and ground-truth/Image Index mapping are not frozen.",
        "num_images": len(output_paths),
        "batch_size": args.batch_size,
        "device": str(device),
        "workers": args.workers,
        "inference_seconds_excluding_model_load": elapsed,
        "images_per_second_excluding_model_load": len(output_paths) / elapsed if elapsed > 0 else None,
    }
    _write_parquet(Path(args.output).expanduser().resolve(), output_paths, scores, output_labels, metadata, pa, pq)
    print(
        f"Wrote {len(output_paths)} rows and {len(output_labels)} finding columns to "
        f"{Path(args.output).expanduser().resolve()}.",
        file=sys.stderr,
    )
    print(
        f"Measured inference throughput: {len(output_paths) / elapsed:.3f} images/s "
        f"({elapsed:.3f}s; model loading excluded). AUROC was not computed.",
        file=sys.stderr,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

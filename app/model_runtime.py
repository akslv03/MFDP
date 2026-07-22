import logging
import os
from typing import Any, Dict, Optional
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from mask_utils import MASK_THRESHOLD, postprocess_mask
from mri_preprocessing import preprocess_volume
from similarity_search import search_similar_cases
from volume_io import (
    is_image_path,
    is_zip_path,
    prepare_volume_from_image,
    prepare_volume_from_zip,
)

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INFERENCE_SIZE = 160
BATCH_SIZE = 8

WEIGHT_CANDIDATES = (
    "unet_transformer_finetuned_best.pt",
    "unet_transformer_finetuned_best.pth",
    "unet_best.pt",
    "unet_best.pth",
    "best_model.pt",
    "best_model.pth",
    "model.pt",
    "model.pth",
    "checkpoint.pt",
    "checkpoint.pth",
)


class MRISegmentationService:
    """Инференс UNet + SegFormer (mit_b2)."""

    def __init__(self, model_path: str | None = None):
        self.weights_dir = os.path.join(os.path.dirname(__file__), "weights")
        self.model_path = model_path or self.weights_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.model = None
        self.loaded_checkpoint: str | None = None

    def _build_model(self):
        model = smp.Unet(
            encoder_name="mit_b2",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )
        model.to(self.device)
        return model

    def _candidate_model_paths(self) -> list[str]:
        if not self.model_path:
            return []

        if os.path.isdir(self.model_path):
            return [os.path.join(self.model_path, name) for name in WEIGHT_CANDIDATES]

        if os.path.isfile(self.model_path):
            return [self.model_path]

        return []

    def _resolve_model_path(self) -> str:
        candidates = self._candidate_model_paths()
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        expected = ", ".join(WEIGHT_CANDIDATES)
        if self.model_path and os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"В папке нет весов модели: {self.model_path}. "
                f"Ожидался один из файлов: {expected}"
            )

        if self.model_path:
            raise FileNotFoundError(
                f"Веса модели не найдены: {self.model_path}. "
                f"Положите чекпоинт в {self.weights_dir} ({expected})."
            )

        raise FileNotFoundError(
            f"Веса модели не найдены. Положите .pt/.pth в {self.weights_dir} "
            f"({expected})."
        )

    def _load_state_dict(self, path: str):
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)

        if isinstance(payload, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value
            if all(isinstance(k, str) for k in payload.keys()):
                return payload

        if hasattr(payload, "state_dict"):
            return payload.state_dict()

        if isinstance(payload, torch.nn.Module):
            return payload.state_dict()

        raise TypeError(f"Неподдерживаемый формат чекпоинта: {type(payload)!r}")

    def _load_model(self):
        if self.model is not None:
            return self.model

        model_path = self._resolve_model_path()
        logger.info("Loading MRI segmentation checkpoint: %s", model_path)

        model = self._build_model()
        state_dict = self._load_state_dict(model_path)

        cleaned_state_dict = {
            key.replace("module.", ""): value for key, value in state_dict.items()
        }

        try:
            model.load_state_dict(cleaned_state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Чекпоинт не совместим с smp.Unet(mit_b2): {model_path}. {exc}"
            ) from exc

        model.eval()
        self.model = model
        self.loaded_checkpoint = model_path
        return model

    def _volume_tensor(self, volume_nhwc: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(volume_nhwc.transpose(0, 3, 1, 2).astype(np.float32))
        return self.normalize(tensor).to(self.device)

    def _weighted_volume_embedding(
        self, model, tensor_nchw: torch.Tensor
    ) -> tuple[list[float], np.ndarray, np.ndarray]:
        """
        Эмбеддинг пациента: среднее эмбеддингов срезов с весами по площади опухоли.
        """
        emb_chunks = []
        mask_weights = []
        prob_chunks = []
        binary_chunks = []

        with torch.no_grad():
            for start in range(0, tensor_nchw.shape[0], BATCH_SIZE):
                batch = tensor_nchw[start : start + BATCH_SIZE]
                pred = model(batch)
                binary = (pred > MASK_THRESHOLD).float()
                mask_weights.append(binary.sum(dim=[1, 2, 3]).cpu().numpy())
                prob_chunks.append(pred.squeeze(1).cpu().numpy())
                binary_chunks.append(binary.squeeze(1).cpu().numpy())
                feats = model.encoder(batch)[-1]
                pooled = F.adaptive_avg_pool2d(feats, (1, 1)).view(batch.size(0), -1)
                emb_chunks.append(pooled.cpu().numpy())

        slice_emb = np.concatenate(emb_chunks, axis=0)
        weights = np.concatenate(mask_weights, axis=0).astype(np.float32)
        probs = np.concatenate(prob_chunks, axis=0)
        binaries = np.concatenate(binary_chunks, axis=0)

        if weights.sum() <= 1e-8:
            weights = np.ones_like(weights)
        else:
            weights = weights + 1e-5
        weights = weights / weights.sum()

        patient_emb = np.average(slice_emb, axis=0, weights=weights).astype(np.float32)
        norm = float(np.linalg.norm(patient_emb))
        if norm > 1e-12:
            patient_emb = patient_emb / norm
        return patient_emb.tolist(), probs, binaries

    def _flair_gray_u8(self, rgb_u8: np.ndarray) -> np.ndarray:
        if rgb_u8.ndim == 2:
            return rgb_u8.astype(np.uint8)
        if rgb_u8.shape[-1] >= 2:
            return rgb_u8[:, :, 1].astype(np.uint8)
        return rgb_u8[:, :, 0].astype(np.uint8)

    def _save_slice_outputs(
        self,
        *,
        out_dir: str,
        base_name: str,
        rgb_u8: np.ndarray,
        mask_prob: np.ndarray,
        name_suffix: str = "",
    ) -> Dict[str, Any]:
        mask = postprocess_mask(mask_prob, threshold=MASK_THRESHOLD)
        mask_u8 = (mask * 255).astype(np.uint8)
        gray = self._flair_gray_u8(rgb_u8)
        stem = f"{base_name}{name_suffix}"

        display_path = os.path.join(out_dir, f"{stem}_display.png")
        mask_path = os.path.join(out_dir, f"{stem}_mask.png")
        overlay_path = os.path.join(out_dir, f"{stem}_overlay.png")

        Image.fromarray(gray, mode="L").save(display_path)
        Image.fromarray(mask_u8, mode="L").save(mask_path)

        overlay = np.stack([gray, gray, gray], axis=-1)
        overlay[mask_u8 > 0] = [255, 0, 0]
        Image.fromarray(overlay).save(overlay_path)

        return {
            "display_image_path": display_path,
            "mask_path": mask_path,
            "overlay_image_path": overlay_path,
            "tumor_ratio": float(np.mean(mask > 0)),
            "tumor_area": float(np.sum(mask > 0)),
        }

    def _search(self, embedding, patient_age, patient_gender) -> str:
        return search_similar_cases(
            embedding,
            age=patient_age,
            gender=patient_gender,
            limit=3,
        )

    def _run_volume(
        self,
        volume: np.ndarray,
        kept_paths,
        *,
        out_dir: str,
        base_name: str,
        patient_age: Optional[int],
        patient_gender: Optional[str],
        input_mode: str,
    ) -> Dict[str, Any]:
        model = self._load_model()
        tensor = self._volume_tensor(volume)
        embedding, probs, binaries = self._weighted_volume_embedding(model, tensor)

        areas = binaries.reshape(binaries.shape[0], -1).sum(axis=1)
        best_i = int(np.argmax(areas)) if areas.size else 0

        slice_gallery: list[dict[str, Any]] = []
        best_outputs: Dict[str, Any] | None = None
        for idx in range(int(volume.shape[0])):
            rgb_u8 = (volume[idx] * 255.0).astype(np.uint8)
            suffix = "" if idx == best_i else f"_s{idx:03d}"
            outputs = self._save_slice_outputs(
                out_dir=out_dir,
                base_name=base_name,
                rgb_u8=rgb_u8,
                mask_prob=probs[idx],
                name_suffix=suffix,
            )
            entry = {
                "index": idx,
                "name": kept_paths[idx].name if idx < len(kept_paths) else f"slice_{idx}",
                "is_best": idx == best_i,
                "tumor_area": float(areas[idx]) if areas.size else 0.0,
                "display_image_path": outputs["display_image_path"],
                "mask_path": outputs["mask_path"],
                "overlay_image_path": outputs["overlay_image_path"],
            }
            slice_gallery.append(entry)
            if idx == best_i:
                best_outputs = outputs

        assert best_outputs is not None
        slice_gallery.sort(
            key=lambda item: (0 if item["is_best"] else 1, -item["tumor_area"], item["index"])
        )

        similarity_cases = self._search(embedding, patient_age, patient_gender)
        return {
            **best_outputs,
            "slice_gallery": slice_gallery,
            "similarity_cases": similarity_cases,
            "checkpoint": self.loaded_checkpoint,
            "threshold": MASK_THRESHOLD,
            "n_slices": int(volume.shape[0]),
            "best_slice_index": best_i,
            "best_slice_name": kept_paths[best_i].name if kept_paths else None,
            "input_mode": input_mode,
        }

    def predict_volume_zip(
        self,
        zip_path: str,
        *,
        patient_age: Optional[int] = None,
        patient_gender: Optional[str] = None,
    ) -> Dict[str, Any]:
        out_dir = os.path.dirname(zip_path)
        base_name = os.path.splitext(os.path.basename(zip_path))[0]
        extract_dir = os.path.join(out_dir, f"{base_name}_vol")

        volume_raw, kept_paths, all_paths = prepare_volume_from_zip(zip_path, extract_dir)
        volume = preprocess_volume(volume_raw, INFERENCE_SIZE)
        logger.info(
            "Volume zip %s: %d files, %d slices used (after dropping edges)",
            zip_path,
            len(all_paths),
            len(kept_paths),
        )

        result = self._run_volume(
            volume,
            kept_paths,
            out_dir=out_dir,
            base_name=base_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            input_mode="volume_zip",
        )
        result["n_input_files"] = len(all_paths)
        return result

    def predict_single_image(
        self,
        image_path: str,
        *,
        patient_age: Optional[int] = None,
        patient_gender: Optional[str] = None,
    ) -> Dict[str, Any]:
        out_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        volume_raw, kept_paths = prepare_volume_from_image(image_path)
        volume = preprocess_volume(volume_raw, INFERENCE_SIZE)
        return self._run_volume(
            volume,
            kept_paths,
            out_dir=out_dir,
            base_name=base_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            input_mode="single_image",
        )

    def predict(
        self,
        image_path: str,
        *,
        patient_age: Optional[int] = None,
        patient_gender: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        if is_zip_path(image_path):
            return self.predict_volume_zip(
                image_path,
                patient_age=patient_age,
                patient_gender=patient_gender,
            )

        if is_image_path(image_path):
            return self.predict_single_image(
                image_path,
                patient_age=patient_age,
                patient_gender=patient_gender,
            )

        raise ValueError(
            "Нужен ZIP-архив со срезами МРТ или изображение (png/tif/jpg/bmp)"
        )

import logging
import os
from typing import Any, Dict, Optional
from model_runtime import MRISegmentationService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

service = MRISegmentationService()


def do_task(
    image_path: str,
    patient_age: Optional[int] = None,
    patient_gender: Optional[str] = None,
) -> Dict[str, Any]:
    """Запускает сегментацию и поиск похожих случаев."""
    try:
        if not os.path.exists(image_path):
            raise RuntimeError(f"Файл изображения не найден: {image_path}")

        result = service.predict(
            image_path,
            patient_age=patient_age,
            patient_gender=patient_gender,
        )
        logger.info("Segmentation completed for %s", image_path)
        return result
    except Exception as e:
        logger.error("Unexpected error during MRI segmentation: %s", e)
        raise RuntimeError(f"Ошибка при сегментации снимка: {str(e)}") from e

"""Скачивает датасет с Kaggle"""

from pathlib import Path
import kagglehub

path = Path(kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation"))
source = path / "kaggle_3m"
target = Path(__file__).resolve().parents[1] / "kaggle_3m"

if target.exists() or target.is_symlink():
    target.unlink()
target.symlink_to(source, target_is_directory=True)

print("Path to dataset files:", target)

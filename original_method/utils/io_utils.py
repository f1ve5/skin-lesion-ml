"""I/O 工具：数据加载、CSV 读写（升级版，兼容中文路径 + 多种 mask 命名）"""
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd


def parse_image_id(filename: str) -> Tuple[str, str]:
    """
    从文件名解析 image_id 和 aug_tag。
      "123_aug1.jpg" -> ("123", "aug1")
      "123.jpg"      -> ("123", "orig")
    """
    name = Path(filename).stem
    m = re.match(r"^(\d+)(?:_aug(\d+))?$", name)
    if m is None:
        return name, "orig"
    base_id = m.group(1)
    aug_tag = f"aug{m.group(2)}" if m.group(2) else "orig"
    return base_id, aug_tag


def load_label_csv(csv_path: str) -> pd.DataFrame:
    """读取标签 CSV，期望两列 image_id / dx"""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "image_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "image_id"})
    if "dx" not in df.columns:
        df = df.rename(columns={df.columns[1]: "dx"})
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["dx"]       = df["dx"].astype(str).str.strip().str.lower()
    return df


def list_image_files(image_dir: str,
                       exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
    files = []
    for fn in os.listdir(image_dir):
        if fn.lower().endswith(exts):
            files.append(fn)
    files.sort(key=lambda x: (parse_image_id(x)[0], parse_image_id(x)[1]))
    return files


def find_mask_path(mask_dir: str, image_filename: str) -> Optional[str]:
    """
    在 mask 文件夹中寻找对应 mask，支持多种常见命名：
      mask_{stem}.{ext}        ← 用户的真实数据用这种
      {stem}.{ext}
      {stem}_mask.{ext}
      {stem}_segmentation.{ext}
    """
    stem = Path(image_filename).stem
    candidates = [
        f"mask_{stem}.jpg", f"mask_{stem}.png", f"mask_{stem}.bmp",
        f"{stem}.png", f"{stem}.jpg", f"{stem}.bmp",
        f"{stem}_mask.png", f"{stem}_mask.jpg",
        f"{stem}_segmentation.png", f"{stem}_segmentation.jpg",
    ]
    for c in candidates:
        p = os.path.join(mask_dir, c)
        if os.path.exists(p):
            return p
    return None


def read_image_bgr(path: str) -> np.ndarray:
    """
    读取彩色图。用 np.fromfile + cv2.imdecode 绕开 OpenCV 对中文路径的限制。
    """
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Cannot read image: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return img


def read_mask(path) -> Optional[np.ndarray]:
    """
    读取 mask 并二值化。path 可为 None 或不存在的路径，
    此时返回 None，让上游用 Otsu 降级。
    """
    if path is None or not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    m = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    _, mb = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return mb


def build_dataset_index(image_dir: str, mask_dir: str,
                          label_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建索引表，每行 = 一张图。
    返回字段: image_id, aug_tag, dx, filename, image_path, mask_path
    没有 mask 的行 mask_path 为 None（上游会用 Otsu 自动分割）。
    """
    label_map = dict(zip(label_df["image_id"].astype(str), label_df["dx"]))
    rows = []
    for fn in list_image_files(image_dir):
        base_id, aug_tag = parse_image_id(fn)
        if base_id not in label_map:
            continue
        rows.append({
            "image_id":   base_id,
            "aug_tag":    aug_tag,
            "filename":   fn,
            "dx":         label_map[base_id],
            "image_path": os.path.join(image_dir, fn),
            "mask_path":  find_mask_path(mask_dir, fn),
        })
    df = pd.DataFrame(rows)
    return df

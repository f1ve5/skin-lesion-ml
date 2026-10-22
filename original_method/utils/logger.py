"""日志工具：所有模块共用"""
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "skin_lesion", log_dir: Path = None) -> logging.Logger:
    """配置统一的日志器，同时输出到文件和控制台"""
    logger = logging.getLogger(name)
    if logger.handlers:  # 避免重复添加handler
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # 控制台输出
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件输出
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"run_{stamp}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

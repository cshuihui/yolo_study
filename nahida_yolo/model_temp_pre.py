"""使用最新训练权重，批量检测数据集 temp 目录中的图片。"""
import re
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
TEMP_DIR = ROOT_DIR / "anime_datasets" / "images" / "temp"
OUTPUT_DIR = TEMP_DIR / "pre"
RUNS_DIR = ROOT_DIR / "runs"
TASK_PREFIX = "nahida_yolo11n"
CONFIDENCE = 0.25
IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def find_latest_weight():
    """选择版本号最大的有效 best.pt；没有权重时提示先训练。"""
    weights = []
    if RUNS_DIR.is_dir():
        for directory in RUNS_DIR.iterdir():
            if not directory.is_dir():
                continue
            match = re.fullmatch(rf"{re.escape(TASK_PREFIX)}_v(\d+)", directory.name)
            version = int(match.group(1)) if match else (1 if directory.name == TASK_PREFIX else None)
            weight = directory / "weights" / "best.pt"
            if version is not None and weight.is_file():
                weights.append((version, weight))
    if not weights:
        raise FileNotFoundError(f"在 {RUNS_DIR} 中找不到训练好的 best.pt，请先训练模型。")
    return max(weights, key=lambda item: item[0])[1]


def main():
    if not TEMP_DIR.is_dir():
        raise FileNotFoundError(f"找不到图片目录: {TEMP_DIR}")
    images = sorted(path for path in TEMP_DIR.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"没有找到图片: {TEMP_DIR}")
        return

    weight = find_latest_weight()
    print(f"使用模型: {weight}\n图片数量: {len(images)}\n结果目录: {OUTPUT_DIR}")
    model = YOLO(str(weight), task="detect")
    # 清空旧结果，避免和本次推理结果混在一起。
    if OUTPUT_DIR.is_symlink() or (OUTPUT_DIR.exists() and OUTPUT_DIR.resolve().parent != TEMP_DIR.resolve()):
        raise RuntimeError(f"输出目录是链接或不在预期位置，拒绝清空: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in OUTPUT_DIR.iterdir():
        if item.is_symlink() or not item.is_dir():
            item.unlink()
        else:
            shutil.rmtree(item)
    print(f"已清空旧结果: {OUTPUT_DIR}")
    failed = []
    for index, image in enumerate(images, 1):
        try:
            result = model.predict(source=str(image), conf=CONFIDENCE, verbose=False)[0]
            output_name = image.name if image.suffix.lower() not in {".dng", ".mpo"} else image.name + ".jpg"
            result.save(filename=str(OUTPUT_DIR / output_name))
            print(f"[{index}/{len(images)}] {image.name}: 检测到 {len(result.boxes)} 个目标")
        except Exception as exc:
            failed.append(image.name)
            print(f"[{index}/{len(images)}] {image.name}: 处理失败: {exc}")
    if failed:
        raise RuntimeError(f"{len(failed)} 张图片处理失败: {', '.join(failed)}")
    print(f"处理完成，结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

import json
import logging
import sys
import traceback
from datetime import datetime
import os
import tempfile
import re
import random
import uuid
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from ultralytics import YOLO

# 确保脚本所在目录为工作基准
ROOT_DIR = Path(__file__).resolve().parent

# =========================================================================
# 核心全局配置
# =========================================================================
MODEL_ORIGINAL_VERSION = "yolo8n"                     # 模型系列标识，用于任务命名；实际模型结构由基础权重决定
MODEL_BASE_VERSION = "anime-yolo8n.pt"                         # 基础权重文件，相对于本脚本目录
BASE_TASK_PREFIX = f"nahida_{MODEL_ORIGINAL_VERSION}"            # 任务名前缀
AUTO_DETECT_VERSION = True                     # 是否开启自动递增版本号与自动寻优权重（True: 推荐全自动; False: 手动）

# 仅在 AUTO_DETECT_VERSION = False 时生效的手动配置：
MANUAL_TASK_NAME = f"nahida_{MODEL_ORIGINAL_VERSION}_v1"
MANUAL_PRETRAINED_WEIGHT = f"runs/{MANUAL_TASK_NAME}/weights/best.pt"

EPOCHS = 40                                     # 训练轮次（针对新增难负样本微调 20~25 轮即可）
BATCH_SIZE = 24                                 # 批次大小（RTX 4060 推荐 16）
LR0 = 0.0005                                    # 初始学习率（二次微调推荐 0.0005 小步调优）
WORKERS = 4                                     # 数据加载多线程数（Windows 推荐 1~4）
DEGREES = 5.0                                  # 随机微倾斜 ±N°（适应歪头卖萌、漂浮构图）

# 可选：从训练集和验证集分别抽样，生成临时小目标样本
ENABLE_SMALL_SAMPLES = True                    # True 启用；False 跳过
SMALL_SAMPLE_RATIO = 0.8                       # 两个集合各自的抽样比例，范围 0~1
SMALL_SCALE_RANGE = (0.3, 0.7)                # 每张图片的随机缩小倍数范围
# =========================================================================




class TeeStream:
    """保留控制台输出，同时将 Python 输出写入日志文件。"""
    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, text):
        self.original.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


@contextmanager
def save_run_log(task_dir: Path):
    """记录训练、验证、导出和清理过程中的 Python 控制台输出。"""
    log_dir = task_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}.log"
    original_out, original_err = sys.stdout, sys.stderr
    with log_path.open("x", encoding="utf-8", buffering=1) as log_file:
        out_tee = TeeStream(original_out, log_file)
        err_tee = TeeStream(original_err, log_file)
        # Ultralytics 在导入时绑定了日志流，需要同步切换已存在的处理器。
        handlers = []
        loggers = [logging.getLogger()] + [logger for logger in logging.root.manager.loggerDict.values()
                                         if isinstance(logger, logging.Logger)]
        for logger in loggers:
            for handler in logger.handlers:
                stream = getattr(handler, "stream", None)
                if stream is original_out or stream is original_err:
                    handler.setStream(out_tee if stream is original_out else err_tee)
                    handlers.append((handler, stream))
        try:
            with redirect_stdout(out_tee), redirect_stderr(err_tee):
                print(f"[日志] 运行日志: {log_path}")
                try:
                    yield
                except BaseException:
                    traceback.print_exc()
                    raise
        finally:
            for handler, stream in handlers:
                handler.setStream(stream)


@contextmanager
def temporary_small_samples(dataset_root: Path):
    """为各分区临时生成小目标样本；只删除本次创建的文件。"""
    if not ENABLE_SMALL_SAMPLES:
        yield
        return
    if not 0 <= SMALL_SAMPLE_RATIO <= 1:
        raise ValueError("SMALL_SAMPLE_RATIO 必须在 0 到 1 之间")
    low, high = SMALL_SCALE_RANGE
    if not 0 < low <= high <= 1:
        raise ValueError("SMALL_SCALE_RANGE 必须满足 0 < 最小值 <= 最大值 <= 1")

    from PIL import Image, ImageOps

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    generated = []
    run_id = uuid.uuid4().hex
    try:
        for split in ("train", "val"):
            image_dir = dataset_root / "images" / split
            label_dir = dataset_root / "labels" / split
            if not image_dir.is_dir() or not label_dir.is_dir():
                raise FileNotFoundError(f"缺少数据集目录: {image_dir} 或 {label_dir}")
            candidates = []
            for image in sorted(image_dir.iterdir()):
                if not image.is_file() or image.suffix.lower() not in extensions:
                    continue
                label = label_dir / (image.stem + ".txt")
                if label.is_file() and label.read_text(encoding="utf-8").strip():
                    candidates.append((image, label))
            selected = random.sample(candidates, int(len(candidates) * SMALL_SAMPLE_RATIO))
            print(f"[小目标] {split}: 有标签 {len(candidates)} 张，抽取 {len(selected)} 张")
            for index, (image_path, label_path) in enumerate(selected):
                with Image.open(image_path) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                    width, height = image.size
                    scale = random.uniform(low, high)
                    scaled_width = max(1, round(width * scale))
                    scaled_height = max(1, round(height * scale))
                    left = random.randint(0, width - scaled_width)
                    top = random.randint(0, height - scaled_height)
                    canvas = Image.new("RGB", (width, height), (114, 114, 114))
                    canvas.paste(image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS), (left, top))
                output_labels = []
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError(f"标签不是 YOLO 检测框格式: {label_path}: {line}")
                    category, x, y, box_width, box_height = parts
                    x, y, box_width, box_height = map(float, (x, y, box_width, box_height))
                    x = (left + x * scaled_width) / width
                    y = (top + y * scaled_height) / height
                    box_width *= scaled_width / width
                    box_height *= scaled_height / height
                    output_labels.append(f"{category} {x:.8f} {y:.8f} {box_width:.8f} {box_height:.8f}")
                stem = f"__small_{run_id}_{index}"
                output_image = image_dir / (stem + ".png")
                output_label = label_dir / (stem + ".txt")
                # 独占创建，确保不会覆盖或删除已有文件。
                with output_image.open("xb") as file:
                    generated.append(output_image)
                    canvas.save(file, format="PNG")
                with output_label.open("x", encoding="utf-8", newline="\n") as file:
                    generated.append(output_label)
                    file.write("\n".join(output_labels) + "\n")
        yield
    finally:
        for path in reversed(generated):
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                print(f"[小目标] 清理失败，请手动删除 {path}: {error}")
        if generated:
            print(f"[小目标] 已尝试清理本次生成的 {len(generated)} 个文件")


def get_task_and_weight(runs_dir: Path, base_name: str):
    """
    自动扫描 runs 目录：
    1. 按 base_name 查找对应系列的历史任务目录及 _v<N> 版本；
    2. 找到当前最高版本号 N，自动将新任务名设为 base_name_v<N+1>；
    3. 自动将最新版本下的 weights/best.pt 作为基础预训练权重；
    4. 无历史权重可用时，加载 MODEL_BASE_VERSION 指定的基础权重。
    """
    if not runs_dir.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)

    versions = []
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name == base_name:
            versions.append((1, p))
        else:
            m = re.match(rf"^{re.escape(base_name)}_v(\d+)$", p.name)
            if m:
                versions.append((int(m.group(1)), p))

    if not versions:
        # 无历史记录，从 v1 和全局配置的基础权重开始
        task_name = f"{base_name}_v1"
        weight_path = ROOT_DIR / MODEL_BASE_VERSION
        print(f"[自动管理] 未检测到历史版本，自动初始化任务: {task_name}")
        print(f"[自动管理] 基础预训练权重: {weight_path}")
        return task_name, weight_path

    # 按版本升序排列
    versions.sort(key=lambda x: x[0])
    latest_ver, latest_dir = versions[-1]
    next_task_name = f"{base_name}_v{latest_ver + 1}"

    # 寻找最近一个包含 best.pt 的有效历史权重
    best_weight = latest_dir / "weights" / "best.pt"
    if not best_weight.exists():
        fallback_weight = None
        for ver, d in reversed(versions):
            w = d / "weights" / "best.pt"
            if w.exists():
                fallback_weight = w
                break
        if fallback_weight:
            print(f"[提示] 最新目录 {latest_dir.name} 未找到 best.pt，已回退至: {fallback_weight}")
            weight_path = fallback_weight
        else:
            weight_path = ROOT_DIR / MODEL_BASE_VERSION
    else:
        weight_path = best_weight

    print(f"[自动管理] 检测到当前最高版本: {latest_dir.name} (v{latest_ver})")
    print(f"[自动管理] 本次自动递增新任务名: {next_task_name}")
    print(f"[自动管理] 本次自动继承最优权重: {weight_path}")
    return next_task_name, weight_path


def select_task_and_weight(runs_dir: Path):
    """先确定任务，以便日志从生成样本开始就写入对应目录。"""
    if AUTO_DETECT_VERSION:
        return get_task_and_weight(runs_dir, BASE_TASK_PREFIX)
    model_weight = ROOT_DIR / MANUAL_PRETRAINED_WEIGHT
    if not model_weight.exists():
        model_weight = ROOT_DIR / MODEL_BASE_VERSION
    return MANUAL_TASK_NAME, model_weight


def main(data_yaml: Path, task_name: str, model_weight: Path):
    runs_dir = ROOT_DIR / "runs"

    # 2. 加载模型权重
    print(f"\n[1/3] 正在加载基础模型权重: {model_weight}")
    model = YOLO(str(model_weight))

    # 3. 配置训练超参数并启动训练
    print(f"[2/3] 开始训练任务 [{task_name}]，数据集配置: {data_yaml}")
    results = model.train(
        # ── 基础路径与设备 ──
        data=str(data_yaml),
        project=str(runs_dir),           # 训练日志与权重输出根目录
        name=task_name,                  # 本次训练任务子目录名称
        exist_ok=True,                   # 覆盖同名训练记录目录，避免堆积 exp1, exp2
        device=0,                        # 使用显卡 0（RTX 4060 8GB）
        workers=WORKERS,                 # 数据加载多线程数（Windows 推荐 1~4）

        # ── 训练轮数与批次 ──
        epochs=EPOCHS,                   # 最大训练轮次
        patience=20,                     # 早停机制：若连续 20 轮验证集指标不上升则提前结束
        batch=BATCH_SIZE,                # Batch Size
        imgsz=640,                       # 输入分辨率（标准 640x640）
        save=True,                       # 保存训练权重文件
        save_period=-1,                  # 关闭周期性 checkpoint
        cache=False,                     # 关闭训练图片缓存

        # ── 优化器与学习率 ──
        optimizer="AdamW",               # 选用 AdamW（小数据集/同人插画迁移收敛极稳）
        lr0=LR0,                         # 初始学习率
        lrf=0.01,                        # 最终学习率衰减比率（最终 lr = lr0 * lrf）
        weight_decay=0.0005,             # 权重衰减系数（防过拟合）
        warmup_epochs=2.0,               # 热身学习率轮数（前 2 轮缓步拉升 lr，防权重崩盘）

        # ── 针对二次元插画特化的数据增强 ──
        fliplr=0.5,                      # 50% 水平镜像翻转（解决发型左/右侧不同画风）
        flipud=0.0,                      # 0% 严禁上下颠倒（动漫角色不会倒立）
        degrees=DEGREES,                 # 随机微倾斜 ±N°（适应歪头卖萌、漂浮构图）
        scale=0.5,                       # 尺度缩放 0.5x ~ 1.5x（平衡大头特写与远景全身）
        translate=0.1,                   # 画面微位移 10%
        mosaic=1.0,                      # 100% 启用 Mosaic 拼接（强化复杂背景和多尺度特征）
        erasing=0.25,                    # 25% 随机局部遮挡（攻克抱膝蹲下、手挡脸、道具遮挡）

        # ── 保护纳西妲色彩特异性的关键参数 ──
        hsv_h=0.015,                     # 色相抖动严格限制在 ±1.5%（保护白绿渐变发与绿瞳不被串色）
        hsv_s=0.4,                       # 饱和度适度抖动 40%（适应水彩淡彩与浓郁厚涂）
        hsv_v=0.3,                       # 明暗度适度抖动 30%（适应黄昏逆光与夜景）
        close_mosaic=8,                  # 最后 8 轮关闭 Mosaic，使用真实完整画风冲刺高精度
    )

    # 4. 训练完成后自动导出为 ONNX 单文件
    best_weight = runs_dir / task_name / "weights" / "best.pt"
    if best_weight.exists():
        print(f"\n[3/3] 训练完成！正在将最优模型 {best_weight} 导出为 ONNX...")
        best_model = YOLO(str(best_weight))
        best_model.export(
            format="onnx",
            imgsz=640,
            dynamic=True,                # 支持动态批次
            simplify=True,               # 自动精简算子
            opset=17,
        )
        print(f"ONNX 导出完成！生成路径: {runs_dir / task_name / 'weights' / 'best.onnx'}")


if __name__ == "__main__":
    task_name, model_weight = select_task_and_weight(ROOT_DIR / "runs")
    with save_run_log(ROOT_DIR / "runs" / task_name):
        print(f"[任务] {task_name}，基础权重: {model_weight}")
        # 临时配置在训练及导出期间保持有效，正常退出或抛出异常时自动清理。
        dataset_config = {
            "path": str(ROOT_DIR / "anime_datasets"),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": {0: "nahida"},
        }
        with tempfile.TemporaryDirectory(prefix="nahida_yolo_") as temp_dir:
            data_yaml = Path(temp_dir) / "dataset.yaml"
            # JSON 是 YAML 的子集，且能正确转义 Windows 路径。
            data_yaml.write_text(json.dumps(dataset_config, ensure_ascii=False, indent=2), encoding="utf-8")
            with temporary_small_samples(ROOT_DIR / "anime_datasets"):
                main(data_yaml, task_name, model_weight)

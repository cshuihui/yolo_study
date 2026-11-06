import os
import re
from pathlib import Path
from ultralytics import YOLO

# 确保脚本所在目录为工作基准
ROOT_DIR = Path(__file__).resolve().parent

# =========================================================================
# 核心全局配置
# =========================================================================
BASE_TASK_PREFIX = "nahida_yolo11n"            # 任务名前缀
AUTO_DETECT_VERSION = True                     # 是否开启自动递增版本号与自动寻优权重（True: 推荐全自动; False: 手动）

# 仅在 AUTO_DETECT_VERSION = False 时生效的手动配置：
MANUAL_TASK_NAME = "nahida_yolo11n_v3"
MANUAL_PRETRAINED_WEIGHT = "runs/nahida_yolo11n_v2/weights/best.pt"

EPOCHS = 25                                     # 训练轮次（针对新增难负样本微调 20~25 轮即可）
BATCH_SIZE = 16                                 # 批次大小（RTX 4060 推荐 16）
LR0 = 0.0005                                    # 初始学习率（二次微调推荐 0.0005 小步调优）
# =========================================================================


def get_task_and_weight(runs_dir: Path, base_name: str):
    """
    自动扫描 runs 目录：
    1. 寻找 nahida_yolo11n 以及 nahida_yolo11n_v<N> 的文件夹；
    2. 找到当前最高版本号 N，自动将新任务名设为 nahida_yolo11n_v<N+1>；
    3. 自动将最新版本下的 weights/best.pt 作为基础预训练权重；
    4. 若尚无历史版本，则从 v1 开始并加载官方骨干 yolo11n.pt。
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
        # 无历史记录，从 v1 官方预训练骨干开始
        task_name = f"{base_name}_v1"
        weight_path = ROOT_DIR / "yolo11n.pt"
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
            weight_path = ROOT_DIR / "yolo11n.pt"
    else:
        weight_path = best_weight

    print(f"[自动管理] 检测到当前最高版本: {latest_dir.name} (v{latest_ver})")
    print(f"[自动管理] 本次自动递增新任务名: {next_task_name}")
    print(f"[自动管理] 本次自动继承最优权重: {weight_path}")
    return next_task_name, weight_path


def main():
    runs_dir = ROOT_DIR / "runs"
    data_yaml = ROOT_DIR / "nahida-yolo11n.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"未找到数据集配置文件: {data_yaml}")

    # 1. 确定本次任务名称与初始权重
    if AUTO_DETECT_VERSION:
        task_name, model_weight = get_task_and_weight(runs_dir, BASE_TASK_PREFIX)
    else:
        task_name = MANUAL_TASK_NAME
        model_weight = ROOT_DIR / MANUAL_PRETRAINED_WEIGHT
        if not model_weight.exists():
            model_weight = ROOT_DIR / "yolo11n.pt"

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
        workers=1,                       # 数据加载多线程数（Windows 推荐 1~4）

        # ── 训练轮数与批次 ──
        epochs=EPOCHS,                   # 最大训练轮次
        patience=20,                     # 早停机制：若连续 20 轮验证集指标不上升则提前结束
        batch=BATCH_SIZE,                # Batch Size
        imgsz=640,                       # 输入分辨率（标准 640x640）
        save=True,                       # 保存训练权重文件
        save_period=10,                  # 每 10 轮保存一次周期性 checkpoint

        # ── 优化器与学习率 ──
        optimizer="AdamW",               # 选用 AdamW（小数据集/同人插画迁移收敛极稳）
        lr0=LR0,                         # 初始学习率
        lrf=0.01,                        # 最终学习率衰减比率（最终 lr = lr0 * lrf）
        weight_decay=0.0005,             # 权重衰减系数（防过拟合）
        warmup_epochs=2.0,               # 热身学习率轮数（前 2 轮缓步拉升 lr，防权重崩盘）

        # ── 针对二次元插画特化的数据增强 ──
        fliplr=0.5,                      # 50% 水平镜像翻转（解决发型左/右侧不同画风）
        flipud=0.0,                      # 0% 严禁上下颠倒（动漫角色不会倒立）
        degrees=12.0,                    # 随机微倾斜 ±12°（适应歪头卖萌、漂浮构图）
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
    main()

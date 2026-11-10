# 纳西妲 YOLO 项目使用说明

本文说明 `nahida_yolo` 目录下各脚本的用途、当前参数和输出位置。路径除特别说明外均相对于本目录；参数值以脚本实际内容为准。

## 环境与目录

使用项目的 Python 环境运行，训练及 `.pt` 推理需要 PyTorch、Ultralytics，视频和屏幕预览需要 OpenCV；屏幕捕捉额外需要 `mss`。ONNX 推理需要 `onnxruntime`、OpenCV 和 NumPy。安装依赖时请使用运行脚本的同一 Python 环境，例如：

```powershell
python -m pip install mss
```

目录中的 `yolo11n.pt` 是训练找不到可用历史权重时的基础模型；`runs/` 是训练输出目录；`anime_datasets/` 是数据集目录。单图示例文件 `112876385_p4.jpg` 等为输入图片。

训练脚本使用的数据集结构：

```text
anime_datasets/
  images/train/       训练图片
  images/val/         验证图片
  images/temp/        批量推理输入
  images/temp/pre/    批量推理输出（每次推理前清空）
  labels/train/       与训练图片同名的 .txt 标签
  labels/val/         与验证图片同名的 .txt 标签
```

检测标签每行格式为 `类别 中心x 中心y 宽 高`，坐标和尺寸相对于图片归一化到 0～1；类别 0 为 `nahida`。训练脚本运行时生成临时数据集配置，数据集根目录为 `anime_datasets`。现存的 `nahida-yolo11n.yaml` 不会被当前训练脚本读取。

## model_train.py：训练与导出

```powershell
python model_train.py
```

脚本加载权重，训练并验证模型，将输出写入 `runs/<任务名>/`；如果存在本次任务的 `weights/best.pt`，会尝试导出 `weights/best.onnx`。输出可能还包括 `last.pt`、`results.csv`、曲线图等。运行日志位于 `runs/<任务名>/train_<时间>_<随机字符>.log`，同时继续显示在控制台；直接由底层原生库写出的部分信息可能不在日志中。

### 顶部配置

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `BASE_TASK_PREFIX` | `nahida_yolo11n` | 自动任务名称的前缀。 |
| `AUTO_DETECT_VERSION` | `True` | 自动寻找最高版本、为新任务递增版本，并选择最近一个存在 `best.pt` 的历史版本；找不到则使用目录下的 `yolo11n.pt`。 |
| `MANUAL_TASK_NAME` | `nahida_yolo11n_v1` | 自动模式关闭时使用的任务名；已有同名目录可能被复用。 |
| `MANUAL_PRETRAINED_WEIGHT` | `runs/nahida_yolo11n_v1/weights/best.pt` | 自动模式关闭时的权重路径；不存在则回退到 `yolo11n.pt`。 |
| `EPOCHS` | `40` | 最大训练轮数，早停可能提前结束。 |
| `BATCH_SIZE` | `24` | 每批图片数；较大值通常占用更多显存。显存不足可调低。 |
| `LR0` | `0.0005` | 初始学习率。 |
| `WORKERS` | `4` | 数据加载工作进程数。 |
| `ENABLE_SMALL_SAMPLES` | `True` | 开启临时小目标样本生成；设为 `False` 时跳过。 |
| `SMALL_SAMPLE_RATIO` | `0.8` | 分别从训练集、验证集中有非空标签的图片抽取 80%；数量向下取整。 |
| `SMALL_SCALE_RANGE` | `(0.3, 0.7)` | 每张抽中图片随机缩小的倍数范围，要求 `0 < 最小值 <= 最大值 <= 1`。 |

小目标样本是将整张原图缩小并放到原尺寸的灰色画布的随机位置，检测框的中心和宽高也会相应换算。训练集、验证集各自抽样，生成文件留在各自分区，文件名前缀为 `__small_`。训练、验证和导出结束或发生普通异常时会删除本次生成的文件；强制结束进程或断电时可能残留。验证集包含临时小目标样本后，其评估范围会变化，比较不同次训练的指标时要注意。

### `model.train()` 参数

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `project` / `name` | `runs` / 任务名 | 输出根目录和本次任务子目录。 |
| `exist_ok` | `True` | 允许复用同名输出目录；手动指定任务名时需留意已有结果。 |
| `device` | `0` | 使用第 0 块 CUDA 显卡。 |
| `workers` | `WORKERS` | 数据加载工作进程数。 |
| `epochs` / `batch` | `EPOCHS` / `BATCH_SIZE` | 最大轮数和批次大小。 |
| `patience` | `20` | 验证效果长时间不改善时的早停等待轮数。 |
| `imgsz` | `640` | 模型输入尺寸；提高可能帮助小目标，但增加计算量和显存占用。 |
| `save` / `save_period` | `True` / `-1` | 保存训练权重；关闭按固定轮次保存周期性权重。 |
| `cache` | `False` | 不将图片数据集缓存到内存或磁盘；这不等同于禁用所有标签索引缓存。 |
| `optimizer` | `AdamW` | 优化器类型。 |
| `lr0` / `lrf` | `LR0` / `0.01` | 初始学习率，以及最终学习率相对于初始值的比例。 |
| `weight_decay` | `0.0005` | 权重衰减。 |
| `warmup_epochs` | `2.0` | 学习率预热轮数。 |

### 数据增强

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `fliplr` / `flipud` | `0.5` / `0.0` | 随机水平 / 垂直翻转的概率。 |
| `degrees` | `12.0` | 随机旋转角度范围，约 ±12°。 |
| `scale` | `0.5` | 训练时随机缩放幅度，约 0.5～1.5 倍；不是抽样概率，也不额外生成一份永久图片。 |
| `translate` | `0.1` | 随机平移幅度。 |
| `mosaic` | `1.0` | Mosaic 生效时的拼图概率。 |
| `close_mosaic` | `8` | 最后 8 轮关闭 Mosaic。 |
| `hsv_h` / `hsv_s` / `hsv_v` | `0.015` / `0.4` / `0.3` | 色相、饱和度和明度的随机扰动幅度。 |
| `erasing` | `0.25` | 脚本传入了此值；项目自带 Ultralytics 配置将其说明为**分类**随机擦除参数，因此不能仅凭该设置认定检测任务启用了随机擦除。 |

导出参数：`format="onnx"` 指定格式，`imgsz=640` 指定输入尺寸，`dynamic=True` 开启动态维度，`simplify=True` 尝试简化模型，`opset=17` 指定 ONNX 算子集版本。导出成功与否还取决于相关依赖。

## model_temp_pre.py：批量图片推理

```powershell
python model_temp_pre.py
```

读取 `anime_datasets/images/temp/` 顶层的支持图片，选择 `runs/nahida_yolo11n_v<N>/weights/best.pt` 中版本号最高的可用权重（也兼容无版本后缀的基础任务），逐张保存带框结果到 `images/temp/pre/`。只扫描顶层，不递归子目录。

**运行前会清空 `pre` 内全部内容（包括子目录），请先保存需要的旧结果。** 无输入图片或加载模型失败时不会进入清空步骤。多数图片保留原名，DNG/MPO 结果名会附加 `.jpg`。逐张失败会记录并在结束时报错。

| 配置 | 当前值 | 作用 |
|---|---:|---|
| `TEMP_DIR` / `OUTPUT_DIR` | `images/temp` / `temp/pre` | 输入、输出路径。 |
| `RUNS_DIR` / `TASK_PREFIX` | `runs` / `nahida_yolo11n` | 自动查找权重的目录和任务前缀。 |
| `CONFIDENCE` | `0.25` | 最低检测置信度；调低可能找到更多目标，也可能增加误检。 |
| `IMAGE_EXTENSIONS` | 脚本内集合 | 被扫描的图片扩展名。 |

## model_video_pre.py：视频推理

```powershell
python model_video_pre.py "D:\视频\example.mp4"
python model_video_pre.py "D:\视频\example.mp4" --conf 0.10 --no-preview
```

不传路径时脚本会提示输入。它通过 `model_temp_pre.py` 的 `find_latest_weight()` 选取最新可用 `best.pt`，逐帧识别并在控制台显示进度、目标数和平均处理帧率。默认显示预览，按 `Q` 提前停止并保存已处理帧。

- `video`：输入视频路径。
- `--conf`：置信度阈值，默认 `0.25`。
- `--no-preview`：不打开预览窗口，仍在终端显示进度。
- 输出到 `runs/video_predictions/`，文件名包含时间；输出使用原视频帧率，**不包含原视频音频**。该目录已加入外层项目的 `.gitignore`。

## model_screen_pre.py：实时屏幕推理

```powershell
python model_screen_pre.py
python model_screen_pre.py --monitor 1 --conf 0.10
```

使用 `mss` 捕获显示器，使用最新可用 `best.pt` 持续识别；单个预览窗口会请求置顶，显示检测框、帧数、目标数和平均处理帧率。按 `Q` 或关闭窗口退出；终端也会输出统计。它目前不保存屏幕录像。

- `--monitor`：显示器编号，默认 `1`；`0` 表示所有显示器组成的区域，可用编号由实际显示器数量决定。
- `--conf`：置信度阈值，默认 `0.25`。

预览窗口位于被捕捉屏幕中时可能被反复截入画面；可将窗口移到另一块不被捕捉的显示器。置顶效果也受窗口系统和全屏应用影响。

## model_yolo_pre.py：单张图片的 PyTorch 推理示例

```powershell
python model_yolo_pre.py
```

此脚本直接使用代码里写死的 `runs/nahida_yolo11n_v2/weights/best.pt` 和 `112876385_p4.jpg`，打印检测框的置信度并显示结果。要换模型或图片，请修改文件里的 `YOLO(...)` 和 `model.predict(source=...)` 路径；`task="detect"` 指定目标检测。它不会像批量、视频、屏幕脚本那样自动寻找最新权重。

## model_onnx_pre.py：ONNX 单图推理示例

```powershell
python model_onnx_pre.py
```

使用 ONNX Runtime、OpenCV 和 NumPy 完成预处理、推理、非极大值抑制及绘框，不依赖 Ultralytics 推理接口。默认读取代码中指定的 `runs/nahida_yolo11n/weights/best.onnx` 和 `112876385_p4.jpg`，需要时修改 `model_path` 与 `source`；输出显示在窗口中，按任意键关闭。路径不一定是最新版本，需先确认文件存在。

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `conf_thresh` | `0.5` | 最低置信度。 |
| `iou_thresh` | `0.45` | NMS 去除重复检测框时的重叠阈值。 |
| `providers` | `CPUExecutionProvider` | 默认使用 CPU；使用 GPU 需要兼容的 ONNX Runtime 与 CUDA 环境。 |
| `input_width` / `input_height` | `640` / `640` | 输入图像缩放及填充尺寸，应与模型兼容。 |
| `result.save(path)` | 默认注释 | 取消注释并设置路径可保存带框图片；`plot()` 返回带框的 BGR 图像。 |

此示例的输出解析假设特定 YOLO 检测模型的输出布局，换成其他导出模型时需要确认输出格式和输入尺寸匹配。

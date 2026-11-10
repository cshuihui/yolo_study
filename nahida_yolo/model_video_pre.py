"""视频逐帧识别：实时预览并保存带检测框的视频（不包含音频）。"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from model_temp_pre import find_latest_weight

ROOT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="对视频进行 YOLO 推理，按 Q 提前结束")
    parser.add_argument("video", nargs="?", help="输入视频路径；不传时运行后输入")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值，默认 0.25")
    parser.add_argument("--no-preview", action="store_true", help="关闭预览窗口，仅在终端显示进度")
    args = parser.parse_args()
    video = Path(args.video or input("请输入视频路径: ").strip().strip('"')).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video}")
    weight = find_latest_weight()
    print(f"模型: {weight}\n输入: {video}", flush=True)
    model = YOLO(str(weight), task="detect")
    capture = cv2.VideoCapture(str(video))
    writer = None
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not 0 < fps < 1000:
            raise RuntimeError("无法获取有效的视频帧率，无法按原速度保存。")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        output_dir = ROOT_DIR / "runs" / "video_predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = output_dir / f"{video.stem}_{stamp}.mp4"
        count = 0
        started = time.perf_counter()
        print(f"输出: {output}\n预览窗口按 Q 可提前结束（已处理的帧仍会保存）。", flush=True)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(source=frame, conf=args.conf, verbose=False)[0]
            annotated = result.plot()
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建输出视频: {output}")
            writer.write(annotated)
            count += 1
            rate = count / max(time.perf_counter() - started, 1e-9)
            progress = f"{count}/{total}" if total > 0 else str(count)
            print(f"\r帧 {progress} | 检测 {len(result.boxes)} 个 | 平均 {rate:.1f} FPS", end="", flush=True)
            if not args.no_preview:
                preview = annotated.copy()
                cv2.putText(preview, f"Frame {progress} | {rate:.1f} FPS", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("YOLO video - press Q to stop", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if count == 0:
            raise RuntimeError("视频未能读取到任何帧。")
        print(f"\n完成，已处理 {count} 帧，输出: {output}（不含原视频音频）", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

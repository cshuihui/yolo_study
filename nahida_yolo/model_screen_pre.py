"""使用 MSS 捕获显示器画面并实时运行 YOLO。"""
import argparse
import time

import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

from model_temp_pre import find_latest_weight


def main():
    parser = argparse.ArgumentParser(description="实时识别屏幕，预览窗口按 Q 退出")
    parser.add_argument("--monitor", type=int, default=1, help="显示器编号，默认 1；0 为所有显示器")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值，默认 0.25")
    args = parser.parse_args()
    weight = find_latest_weight()
    print(f"模型: {weight}\n预览窗口按 Q 退出。", flush=True)
    model = YOLO(str(weight), task="detect")
    window_name = "YOLO screen - press Q to stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    try:
        with mss() as screen:
            if args.monitor < 0 or args.monitor >= len(screen.monitors):
                raise ValueError(f"显示器编号无效，可用范围: 0 到 {len(screen.monitors) - 1}")
            monitor = screen.monitors[args.monitor]
            print(f"正在捕捉显示器 {args.monitor}: {monitor}", flush=True)
            count = 0
            started = time.perf_counter()
            while True:
                frame = cv2.cvtColor(np.asarray(screen.grab(monitor)), cv2.COLOR_BGRA2BGR)
                result = model.predict(source=frame, conf=args.conf, verbose=False)[0]
                annotated = result.plot()
                count += 1
                rate = count / max(time.perf_counter() - started, 1e-9)
                cv2.putText(annotated, f"Frame {count} | {len(result.boxes)} detections | {rate:.1f} FPS",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"\r帧 {count} | 检测 {len(result.boxes)} 个 | 平均 {rate:.1f} FPS", end="", flush=True)
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cv2.destroyAllWindows()
        print("\n屏幕推理已结束。", flush=True)


if __name__ == "__main__":
    main()

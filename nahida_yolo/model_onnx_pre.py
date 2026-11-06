import os
import ast
import cv2
import numpy as np
import onnxruntime as ort

root_path = os.path.dirname(os.path.abspath(__file__))
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


class YOLO11ONNX:
    """基于纯 ONNXRuntime 的 YOLO11 目标检测器（不依赖 Ultralytics/PyTorch）"""

    def __init__(self, model_path: str, conf_thresh: float = 0.5, iou_thresh: float = 0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # 优先使用 CPUExecutionProvider，稳定无外部 CUDA dll 缺失困扰
        # 若已配置好完整的 CUDA12 与 cuDNN9 环境，可改为 ['CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        # 读取输入规格
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        
        # 从模型元数据中读取类别名称与输入尺寸（若存在）
        metadata = self.session.get_modelmeta().custom_metadata_map
        if 'names' in metadata:
            try:
                self.classes = ast.literal_eval(metadata['names'])
            except Exception:
                self.classes = {0: 'nahida'}
        else:
            self.classes = {0: 'nahida'}

        self.input_width = 640
        self.input_height = 640

    def letterbox(self, img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
        """等比例缩放图片并填充黑/灰边（Letterbox）"""
        shape = img.shape[:2]  # [h, w]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (new_shape[1] - new_unpad[0]) / 2.0
        dh = (new_shape[0] - new_unpad[1]) / 2.0

        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img_padded, r, (dw, dh)

    def preprocess(self, img: np.ndarray):
        """前处理：Letterbox -> BGR转RGB -> 归一化 -> 转换为 NCHW 格式"""
        img_padded, ratio, (dw, dh) = self.letterbox(img, (self.input_height, self.input_width))
        blob = img_padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)  # (1, 3, 640, 640)
        return blob, ratio, (dw, dh)

    def predict(self, source: str):
        """推理核心流程：读取图片 -> 前处理 -> ONNX推理 -> NMS后处理 -> 坐标还原"""
        img0 = cv2.imread(source)
        if img0 is None:
            raise FileNotFoundError(f"未找到输入图片: {source}")

        # 1. 前处理
        blob, ratio, (dw, dh) = self.preprocess(img0)

        # 2. ONNXRuntime 推理
        outputs = self.session.run(None, {self.input_name: blob})[0]  # shape: (1, 4 + num_classes, 8400)

        # 3. 后处理：转置为 (8400, 4 + num_classes)
        preds = outputs[0].T  # 每行: [cx, cy, w, h, class0_score, class1_score, ...]
        boxes, confidences, class_ids = [], [], []

        num_classes = len(self.classes)
        for row in preds:
            cx, cy, w, h = row[0:4]
            scores = row[4:4 + num_classes]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])

            if conf >= self.conf_thresh:
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                boxes.append([int(x1), int(y1), int(w), int(h)])
                confidences.append(conf)
                class_ids.append(class_id)

        # 4. 非极大值抑制 (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.iou_thresh)

        # 5. 还原坐标至原图尺寸并组织返回结果
        detections = []
        annotated_img = img0.copy()

        if len(indices) > 0:
            for idx in indices:
                i = idx if isinstance(idx, (int, np.integer)) else idx[0]
                box = boxes[i]
                conf = confidences[i]
                cid = class_ids[i]
                cname = self.classes.get(cid, str(cid))

                # 逆变换回原图坐标
                x = (box[0] - dw) / ratio
                y = (box[1] - dh) / ratio
                w = box[2] / ratio
                h = box[3] / ratio

                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(img0.shape[1], int(x + w))
                y2 = min(img0.shape[0], int(y + h))

                detections.append({
                    'class_id': cid,
                    'class_name': cname,
                    'confidence': conf,
                    'box': [x1, y1, x2, y2]
                })

                # 在图上画框和标签
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{cname} {conf:.2f}"
                cv2.putText(annotated_img, label_text, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return ONNXResult(img0, annotated_img, detections)


class ONNXResult:
    """包装检测结果对象，提供类似 Ultralytics Result 的便捷操作接口"""

    def __init__(self, orig_img: np.ndarray, annotated_img: np.ndarray, detections: list):
        self.orig_img = orig_img
        self.annotated_img = annotated_img
        self.detections = detections

    def plot(self):
        """返回已画框的图片 (BGR 格式)"""
        return self.annotated_img

    def show(self, title: str = "ONNX Detection Result"):
        """使用 OpenCV 显示检测结果窗口"""
        cv2.imshow(title, self.annotated_img)
        print("按任意键关闭图片窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, save_path: str):
        """保存画框后的图片"""
        cv2.imwrite(save_path, self.annotated_img)
        print(f"已保存结果图片至: {save_path}")

    def __repr__(self):
        return (f"<ONNXResult: {len(self.detections)} detections, "
                f"detections={self.detections}>")


# =========================================================================
# 运行主流程（完全脱离 Ultralytics，仅使用 onnxruntime + opencv）
# =========================================================================
if __name__ == '__main__':
    # 1. 加载 ONNX 模型
    model = YOLO11ONNX(
        model_path=os.path.join(root_path, './runs/nahida_yolo11n/weights/best.onnx'),
        conf_thresh=0.5,
        iou_thresh=0.45
    )

    # 2. 预测指定图片
    result = model.predict(source=os.path.join(root_path, '112876385_p4.jpg'))

    # 可选：保存结果图片 (类似 model_yolo_pre.py 注释行)
    # result.save(os.path.join(root_path, 'result_onnx.jpg'))

    # 3. 打印预测结果
    print(result)

    # 4. 弹出窗口展示结果（类似 result[0].show()）  
    result.show()

from ultralytics import YOLO

model = YOLO('best_d.pt')

model.export(format='onnx')
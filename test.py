from ultralytics import YOLO
import matplotlib
matplotlib.use('TkAgg')


model = YOLO('yolo11n.pt')
results = model('example.jpg', verbose=False)  # 是否打印结果 
print(results)
results[0].show()   
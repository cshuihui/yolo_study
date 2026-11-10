from ultralytics import YOLO
import os
root_path = os.path.dirname(os.path.abspath(__file__))
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

model = YOLO(os.path.join(root_path, './runs/nahida_yolo11n_v2/weights/best.pt'), task='detect')

# result = model.predict(source=os.path.join(root_path, '112876385_p4.jpg'))
result = model.predict(source=os.path.join(root_path, '112876385_p4.jpg'))
# plt.imsave('result.jpg', result[0].plot()[:, :, ::-1])
print(result[0].boxes.conf)
result[0].show()
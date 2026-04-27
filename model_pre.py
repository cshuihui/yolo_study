from ultralytics import YOLO
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

model = YOLO('best.pt', task='pose')

result = model.predict(source='exg.png')
# plt.imsave('result.jpg', result[0].plot()[:, :, ::-1])
print(result)
result[0].show()
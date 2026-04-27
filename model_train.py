from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')

model.train(data='mymodel-yolo11-pose.yaml',
            workers=0,
            epochs=500,
            batch=32,
            lr0=0.0001,
            pose=40,
            box=5,


            scale=0.3,
            translate=0.1,



            mosaic=0,
            mixup=0,
            copy_paste=0
            )
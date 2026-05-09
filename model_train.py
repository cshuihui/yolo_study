from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')

model.train(data='mymodel-yolo11-pose.yaml',
            workers=0,
            epochs=200,
            batch=32,
            lr0=0.0001,
            lrf=0.01,
            optimizer='AdamW',

            pose=40,
            box=15,


            scale=0.3,
            translate=0.1,
            mosaic=0,
            mixup=0,
            copy_paste=0
            )
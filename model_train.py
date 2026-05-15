from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.train(data='ui_detection_yolo11.yaml',
            workers=0,
            epochs=100,
            batch=32,
            lr0=0.001,
            lrf=0.01,
            optimizer='AdamW',
            weight_decay=0.0005,       # 权重衰减

            box=25,


            scale=0.03,  # 随机缩放增强
            translate=0.1,  # 随机平移增强
            mosaic=0,   # 马赛克
            mixup=0,    # 混合
            copy_paste=0    # 复制粘贴
            )
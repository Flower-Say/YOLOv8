from ultralytics import YOLO

#加载模型
model = YOLO("D:/PyCharm/Project\YOLOv8/ultralytics/models/v8/yolov8m.yaml")
model = YOLO("D:/PyCharm/Project/YOLOv8/yolov8n.pt")

#使用模型
model.info()  # 显示模型信息
model.train(data="coco128.yaml", workers=0, epochs=100)  # 训练模型


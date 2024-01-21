from ultralytics import YOLO

# model = YOLO("runs/segment/train4/weights/best.pt")

model = YOLO("runs/segment/train4/weights/best.onnx")

model.predict(source = "dataset/unseen/7.jpeg",
              show=True,
              save=True,
              hide_labels=False,
              hide_conf=False,
              conf=0.5,
              save_txt=True,
              save_crop=True,
              line_thickness=2)
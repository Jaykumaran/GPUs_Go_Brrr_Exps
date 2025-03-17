
**Torch**
In [1]: `from ultralytics import YOLO`
In [1]:`from ultralytics import YOLO`
In [3]:` results = model.predict("/home/jaykumaran/Downloads/3191251-uhd_4096_2160_25fps.mp4")`


`video 1/1 (frame 327/327) /home/jaykumaran/Downloads/3191251-uhd_4096_2160_25fps.mp4: 352x640 1 person, 1 dog, 3.0ms`
Speed: 1.7ms preprocess, 2.9ms inference, 0.7ms postprocess per image at shape (1, 3, 352, 640)



**TensorRT**
`model = YOLO('yolov8n.pt')`
`model.export(format = "engine", device = 0, save = True)`
`model = YOLO('yolov8n.engine')`
`model = YOLO('yolov8n.engine', task = "detect")`
`Speed: 2.6ms preprocess, 2.3ms inference, 0.9ms postprocess per image at shape (1, 3, 640, 640)`


**ONNX**
`Speed: 2.6ms preprocess, 5.4ms inference, 0.9ms postprocess per image at shape (1, 3, 640, 640)`


**OpenVINO**  --> (Might be something wrong)
`model.export(format = "openvino", device = 'cpu', save = True)`
`model = YOLO('yolov8n_openvino_model/', task = 'detect')`

`Speed: 2.1ms preprocess, 34.6ms inference, 0.8ms postprocess per image at shape (1, 3, 640, 640)`


**Torchscript**
`model.export(format = "torchscript", device = 0 , save = True)`
`model = YOLO('yolov8n.torchscript', task = 'detect')`
`Speed: 1.7ms preprocess, 3.0ms inference, 1.0ms postprocess per image at shape (1, 3, 640, 640)`

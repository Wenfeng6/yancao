import torch
    
device_1 = torch.device('cuda:1')
net_detect_car_1_cuda1 = torch.hub.load('ultralytics/yolov5', 'yolov5l', device=device_1)
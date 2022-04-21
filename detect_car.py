
from tongan.models.experimental import attempt_load
from tongan.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
import cv2
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')

from PIL import Image
from car_attribute.VehicleDC import Car_Classifier

# def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
#     # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
#     shape = img.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)

#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better test mAP)
#         r = min(r, 1.0)

#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

#     dw /= 2  # divide padding into 2 sides
#     dh /= 2

#     if shape[::-1] != new_unpad:  # resize
#         img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return img, ratio, (dw, dh)

# def detection_cars(img_path):
#     model = attempt_load('./tongan/model_bin/tongan8_v3_300epoch.pt', map_location=None).cuda()
#     model.eval()
    
#     # ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'face', 'plate']
#     names = model.module.names if hasattr(model, 'module') else model.names
#     imgsz = 416
#     imgsz = check_img_size(imgsz, s=model.stride.max())

#     if isinstance(img_path, str):
#         img = cv2.imread(img_path)
#     else:
#         img = img_path
    
#     img0_size = img.shape
#     img0 = img.copy()
#     img = letterbox(img, new_shape=imgsz)[0]
#     img = img[:, :, ::-1].transpose(2, 0, 1)
#     img = np.ascontiguousarray(img)

#     img = torch.from_numpy(img).cuda().float()
#     img /= 255.0
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)

#     with torch.no_grad():
#         pred = model(img, augment=False) #元组。0维[1，3780，13]tensor,剩下是list
#         pred = pred[0]
#                             #  pred  conf_thres iou_thres 
#     print(pred.shape)
#     pred = non_max_suppression(pred, 0.4,       0.6,      classes=[2, 4, 5], agnostic=False)
#     if pred[0] != None:
#         pred = pred[0].cpu().numpy()
#     else:
#         pred = []
#     # print(len(pred[0]))
#     # print(pred.shape)
#     print(pred)

#     for i in range(len(pred)):
#         pred[i, 0] = pred[i, 0] / img.shape[3] * img0_size[1]
#         pred[i, 1] = pred[i, 1] / img.shape[2] * img0_size[0]
#         pred[i, 2] = pred[i, 2] / img.shape[3] * img0_size[1]
#         pred[i, 3] = pred[i, 3] / img.shape[2] * img0_size[0]
    
#     if len(pred):
#         xyxy = (int(pred[0, 0]), int(pred[0, 1]), int(pred[0, 2]), int(pred[0, 3]) )
#         plot_one_box(xyxy, img0,
#                     color=[255,0,0], line_thickness=1)
#         cv2.imwrite('img0.png',img0)

#     bboxs = []
#     for i in range(len(pred)):
#         bboxs.append([int(pred[i, 0]), int(pred[i, 1]),
#                      int(pred[i, 2] - pred[i, 0]), int(1.2*(pred[i, 3] - pred[i, 1]))])
#         print(names[int(pred[i, 5])])
        
#     return bboxs # top right bottom left


def detection_cars(img_path, net): #or file, Path, PIL, OpenCV, numpy, list

    # Model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # or yolov5m, yolov5l, yolov5x, custom
    # print(-1)
    model = net[0]

    # classifier = Car_Classifier(
    #     num_cls=19, model_path='./car_attribute/checkpoints/epoch_39.pth')
    # print(-2)
    classifier = net[1]

    # Inference
    # print(-3)
    with torch.no_grad():
        results = model(img_path)

    # import cv2
    # img = cv2.imread(img_path)    

    # print(results.names) 或者用这个输出全部，自己瞅瞅
    # [(x1+x2)/2, (y1+y2)/2, w, h, label]
    # 0 - person
    # 1 - bicycle
    # 2 - car
    # 3 - motorcycle
    # 5 - bus
    # 7 - truck

    # print(-4)
    carname = {'saloonCar':1, 'suv':1, 'van':1, 'passengerCar':2, 'truck':3, 'waggon':3}
    colorname = {'Red':'红色', 'Yellow':'黄色', 'Blue':'蓝色', 'Black':'黑色',
        'Green':'绿色', 'Brown':'棕色', 'White':'白色', 'Gray':'灰色', 'Purple':'紫色'}

    bboxs = []
    H, W, _ = results.imgs[0].shape
    raw_img = results.imgs[0].copy()
    results = results.xywh[0]
    results = results.cpu().data.tolist()
    batch = []
    car_bboxs = []
    # print(-5)
    for result in results:
        # print(-6)
        label = int(result[5])
        if label not in [1, 2, 3, 5, 7]:
            continue
        x = max(0, int(result[0] - result[2] / 2))
        y = max(0, int(result[1] - result[3] / 2))
        w = min(int(result[2]), W - x - 1)
        h = min(int(result[3]), H - y - 1)
        if label == 1:
            bboxs.append([x, y, w, h, 5, ''])
            continue
        elif label == 3:
            bboxs.append([x, y, w, h, 4, ''])
            continue

        img = Image.fromarray(raw_img[y:y+h, x:x+w, :])
        batch.append(img)
        car_bboxs.append([x, y, w, h, -1, ''])

    if len(batch) > 0:
        anss = classifier.predict(batch)

        for i, ans in enumerate(anss):
            # print(-7)
            # ans = classifier.predict(img)
            # ans = list(ans)
            if ans[2] in carname.keys():
                ans[2] = carname[ans[2]]
            if ans[0] in colorname.keys():
                ans[0] = colorname[ans[0]]
            car_bboxs[i][-2] = ans[2]
            car_bboxs[i][-1] = ans[0]
        bboxs = bboxs + car_bboxs

    return bboxs


if __name__=='__main__':
    from car_attribute.VehicleDC import Car_Classifier
    from plate_detector.data import cfg_mnet
    from plate_detector.models.retina import Retina
    from detect_plate import load_model as plate_load_model
    import torch.backends.cudnn as cudnn
    net_detect_car_1 = torch.hub.load('ultralytics/yolov5', 'yolov5l')
    net_detect_car_2 = Car_Classifier(
        num_cls=19, model_path='./car_attribute/checkpoints/epoch_39.pth')
    net_detect_car = [net_detect_car_1, net_detect_car_2]

    location = detection_cars('samples/testcar5.jpg', net_detect_car)
    print(location)
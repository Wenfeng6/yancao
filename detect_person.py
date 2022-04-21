
from tongan.models.experimental import attempt_load
from tongan.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
import cv2
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# # tongan model
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

# def detection_persons(img_path):
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
#     pred = non_max_suppression(pred, 0.4,       0.6,      classes=[0], agnostic=False)
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
    
#         xyxy = (int(pred[i, 0]), int(pred[i, 1]), int(pred[i, 2]), int(pred[i, 3]))
#         plot_one_box(xyxy, img0,
#                     color=[255, 0, 0], line_thickness=1)
#         cv2.imwrite('crop_detect_person.png', img0)

#         # from detect_face import detection_faces
#         # img_person = img0[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
#         # print("detect face from person crop")
#         # print(detection_faces(img_person))

#     bboxs = []
#     for i in range(len(pred)):
#         bboxs.append([int(pred[i, 0]), int(pred[i, 1]), 
#             int(pred[i, 2]) - int(pred[i, 0]), int(pred[i, 3]) - int(pred[i, 1])])
#         # print(names[int(pred[i, 5])])
        
#     return bboxs # top right bottom left

# yolov5I
def detection_persons(img_path, model): #or file, Path, PIL, OpenCV, numpy, list

    # Model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # or yolov5m, yolov5l, yolov5x, custom
    # model.eval()

    # Inference
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

    bboxs = []
    H, W, _ = results.imgs[0].shape
    results = results.xywh[0]
    results = results.cpu().data.tolist()
    for result in results:
        label = int(result[5])
        if label not in [0]:
            continue
        x = max(0, int(result[0] - result[2] / 2))
        y = max(0, int(result[1] - result[3] / 2))
        w = min(int(result[2]), W - x - 1)
        h = min(int(result[3]), H - y - 1)
        bboxs.append([x, y, w, h])
        print([x, y, w, h])

    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
    # cv2.imwrite('crop_anspersons.png', img)
    return bboxs

if __name__=='__main__':
    location = detection_persons('./samples/testperson.png')
    print(location)
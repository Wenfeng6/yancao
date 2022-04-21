
from car_attribute.VehicleDC import Car_Classifier
from PIL import Image
from tongan.models.experimental import attempt_load
from tongan.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
import cv2
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')


def detection_object(img_path, net):  # or file, Path, PIL, OpenCV, numpy, list

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
    carname = {'saloonCar': 1, 'suv': 1, 'van': 1,
               'passengerCar': 2, 'truck': 3, 'waggon': 3}
    colorname = {'Red': '红色', 'Yellow': '黄色', 'Blue': '蓝色', 'Black': '黑色',
                 'Green': '绿色', 'Brown': '棕色', 'White': '白色', 'Gray': '灰色', 'Purple': '紫色'}

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
        if label not in [0, 1, 2, 3, 5, 7]:
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
        elif label == 0:
            bboxs.append([x, y, w, h, 0, ''])
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


if __name__ == '__main__':
    from car_attribute.VehicleDC import Car_Classifier
    from plate_detector.data import cfg_mnet
    from plate_detector.models.retina import Retina
    from detect_plate import load_model as plate_load_model
    import torch.backends.cudnn as cudnn
    net_detect_car_1 = torch.hub.load('ultralytics/yolov5', 'yolov5l')
    net_detect_car_2 = Car_Classifier(
        num_cls=19, model_path='./car_attribute/checkpoints/epoch_39.pth')
    net_detect_car = [net_detect_car_1, net_detect_car_2]

    location = detection_object('samples/testcar5.jpg', net_detect_car)
    print(location)

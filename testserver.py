import numpy as np
import json
import os
import cv2
from flask import Flask, request, jsonify
from gevent import pywsgi
from gevent import monkey
import torch

from extraction import extraction
from compare import match
from detect_face import detection_faces
from detect_person import detection_persons
from detect_car import detection_cars
from detect_plate import detection_plates
from detect_object import detection_object
from recog_plate import get_license
from detect_time import get_time
from attribute_extractor import attribute_extractor

from arcface_torch.backbones import get_model

monkey.patch_all()
app = Flask(__name__)
app.debug = True

# 1.1 获取人脸特征信息
@app.route('/feature_extraction/face', methods=['post'])
def predict():
    data = json.loads(request.get_data().decode('utf-8'))

    if True:
    # try:
        pic_id = data["pictureId"]
        f_path = data["screenshot"]
        feature, is_save = extraction([f_path], net_extraction)
        feature = feature[0]
        str_feature = ""
        for index in range(feature.shape[1]):
            str_feature += str(feature[0, index]) + " "

        result = {
            "code": 200,
            "message": {
                "pictureId": pic_id,
                "version": model_version,
                "feature": str_feature,
                "is_save":is_save
            }
        }
        # print(result)
    # except:
    #     result = {
    #         "code": 500,
    #         "message": "程序异常"
    #     }

    return jsonify(result)

# 1.2 特征对比
@app.route('/feature_comparison', methods=['post'])
def work():
    data = json.loads(request.get_data().decode('utf-8'))

    try:
        query = data["feature"]
        gallery = data["baseFeatures"]
        try:
            mode = data["isMode"]
        except:
            mode = 2

        if isinstance(query, list):
            query = query[0]
        query = get_feature(query)

        gallery_features = []
        for sample in gallery:
            gallery_features.append(get_feature(sample["feature"]))

        similarity = match(gallery_features, query, mode=mode)
        print(f'simi:{similarity}')

        result = {
            "code": 200,
            "message": [{
                "pictureId": gallery[i]["pictureId"],
                "version": model_version,
                "matched": "{}%".format(int(similarity[i] * 100)),
            } for i in range(len(gallery_features))]
        }
    except:
        result = {
            "code": 500,
            "message": "程序异常"
        }

    return jsonify(result)


def get_feature(x):
    x = x.split(' ')
    feature = []
    for num in x[:-1]:
        feature.append(float(num))
    feature = np.array(feature)
    return feature


# 1.3 提取图片目标特写区域（人脸）
@app.route('/target_locate/face', methods=['post'])
def locateFace():
    data = json.loads(request.get_data().decode('utf-8'))

    try:
        pictureId = data["pictureId"]
        picname = data["screenshot"]

        assert os.path.exists(picname)
        
        face_locs = detection_faces(picname, net_detect_face)
        person_locs = detection_persons(picname, net_detection_persons)
        face_person = []

        # 脸找人
        for f in range(len(face_locs)):
            x = face_locs[f][0] + face_locs[f][2] / 2
            y = face_locs[f][1] + face_locs[f][3] / 2
            flag = False
            for p in range(len(person_locs)):              
                person_x_left = person_locs[p][0]
                person_x_right = person_locs[p][0] + person_locs[p][2]
                person_y_left = person_locs[p][1]
                person_y_right = person_locs[p][1] + person_locs[p][3]
                if person_x_left < x < person_x_right and person_y_left < y < person_y_right:
                    face_person.append(person_locs[p])
                    del person_locs[p]
                    flag = True
                    break
            if not flag:
                face_person.append(tuple((0, 0, 0, 0)))
        # 只有人没有脸
        for p in range(len(person_locs)):
            face_person.append(person_locs[p])
            face_locs.append(tuple((0, 0, 0, 0)))
        
        # import cv2
        # img = cv2.imread(picname)
        # for i in range(len(face_locs)):
        #     cv2.rectangle(img, (face_locs[i][0], face_locs[i][1]), \
        #     (face_locs[i][0]+face_locs[i][2], face_locs[i][1]+face_locs[i][3]), (0,0,255), 2)
        #     cv2.rectangle(img, (face_person[i][0], face_person[i][1]), \
        #     (face_person[i][0]+face_person[i][2], face_person[i][1]+face_person[i][3]), (255,0,0), 2)
        # cv2.imwrite('crop_persons.png', img)

        result = {
            "code": 200,
            "message":{
                "pictureId": pictureId,
                "targetList": [
                    {
                        "body": {
                            "x": face_person[i][0],
                            "y": face_person[i][1],
                            "width": face_person[i][2],
                            "height": face_person[i][3]
                        },
                        "face": {
                            "x": face_locs[i][0],
                            "y": face_locs[i][1],
                            "width": face_locs[i][2],
                            "height": face_locs[i][3]
                        }       
                    }for i in range(len(face_locs))
                ]
            }
        }
    except:
        result = {
            "code": 500,
            "message": "程序异常"
        }
    
    return jsonify(result)

# 1.4 提取图片目标特写区域（机动车、非机动车、车牌）
@app.route('/target_locate/vehicle', methods=['post'])
def locateVehicle():
    data = json.loads(request.get_data().decode('utf-8'))

    try:
        pictureId = data["pictureId"]
        picname = data["screenshot"]
        assert os.path.exists(picname)

        car_locs = detection_cars(picname, net_detect_car)
        if len(car_locs) != 0:
            plate_locs = detection_plates(picname, net_detect_plate)
        else:
            plate_locs = []
        car_plate = []

        # 车找车牌
        for c in range(len(car_locs)):
            car_x_left = car_locs[c][0]
            car_x_right = car_locs[c][0] + car_locs[c][2]
            car_y_left = car_locs[c][1]
            car_y_right = car_locs[c][1] + car_locs[c][3]

            flag = False
            for p in range(len(plate_locs)):
                x = plate_locs[p][0]
                y = plate_locs[p][1]
                if car_x_left < x < car_x_right and car_y_left < y < car_y_right:
                    car_plate.append(plate_locs[p])
                    flag = True
                    del plate_locs[p]
                    break
            if not flag:
                car_plate.append(tuple((0, 0, 0, 0)))

        # # 只有车牌没有车
        # for p in range(len(plate_locs)):
        #     car_plate.append(plate_locs[p])
        #     car_locs.append(tuple((0, 0, 0, 0, '', '')))

        # import cv2
        # img = cv2.imread(picname)
        # for i in range(len(car_locs)):
        #     cv2.rectangle(img, (car_locs[i][0], car_locs[i][1]),
        #                   (car_locs[i][0]+car_locs[i][2], car_locs[i][1]+car_locs[i][3]), (255, 0, 0), 2)
        #     cv2.rectangle(img, (car_plate[i][0], car_plate[i][1]),
        #                   (car_plate[i][0]+car_plate[i][2], car_plate[i][1]+car_plate[i][3]), (0, 0, 255), 2)
        #     if car_plate[i][1]:
        #         imgcrop = img[car_plate[i][1]:car_plate[i][1]+car_plate[i][3],
        #                       car_plate[i][0]:car_plate[i][0]+car_plate[i][2]]
        #         cv2.imwrite(f'samples/crop_plate{i}.png', imgcrop)
        # cv2.imwrite('crop_cars.png', img)

        result = {
            "code": 200,
            "message": {
                "pictureId": pictureId,
                "targetList": [
                    {
                        "vehicleClassType": car_locs[i][4],
                        "vehicleColor": car_locs[i][5],
                        "vehicle": {
                            "x": car_locs[i][0],
                            "y": car_locs[i][1],
                            "width": car_locs[i][2],
                            "height": car_locs[i][3],
                        },
                        "plate": {
                            "x": car_plate[i][0],
                            "y": car_plate[i][1],
                            "width": car_plate[i][2],
                            "height": car_plate[i][3]
                        }
                    }for i in range(len(car_locs))
                ]
            }
        }
    except:
        result = {
            "code": 500,
            "message": "程序异常"
        }

    return jsonify(result)


# 1.5 识别车牌信息
@app.route('/get_platemess', methods=['post'])
def get_platemess():
    data = json.loads(request.get_data().decode('utf-8'))

    try:
        pictureId = data["pictureId"]
        picname = data["screenshot"]
        assert os.path.exists(picname)

        license, color, plateReliability, plateCharReliability = get_license(
            picname, model_recog_plate)
        
        result = {
            "code": 200,
            "message": {
                "pictureId": pictureId,
                "plateNo": license,
                "plateColor": color,
                "plateReliability": plateReliability,
                "plateCharReliability": plateCharReliability
            }
        }
        print(result)
    except:
        result = {
            "code": 500,
            "message": "程序异常"
        }
    
    return jsonify(result)


# 1.6 识别车辆属性
@app.route('/get_vehicletype', methods=['post'])
def get_vehicletype():
    data = json.loads(request.get_data().decode('utf-8'))

    try:
        pictureId = data["pictureId"]
        picname = data["screenshot"]
        assert os.path.exists(picname)

        res = attribute_extractor(picname, net_attribute_car)
        if not res:
            vehicletype = ""
            vehiclecolor = ""
        else:
            vehicletype = res[2]
            if len(res[0]) != 0:
                vehiclecolor = res[0]
            else:
                vehiclecolor = ""

        result = {
            "code": 200,
            "message": {
                "pictureId": pictureId,
                "vehicleClassType": vehicletype,
                "vehicleColor": vehiclecolor
            }
        }
    except:
        result = {
            "code": 500,
            "message": "程序异常"
        }
    return jsonify(result)


# 1.7 图像识别监控时间
@app.route('/identify_monitortime', methods=['post'])
def get_monitortime():
    data = json.loads(request.get_data().decode('utf-8'))

    try:
        pictureId = data["pictureId"]
        picname = data["screenshot"]
        assert os.path.exists(picname)

        current_time = get_time(picname, 0)
        print(current_time)
        result = {
            "code": 200,
            "message": {
                "pictureId": pictureId,
                "time": current_time
            }         
        }
    except:
        result = {
            "code": 500,
            "message": "程序异常"
        }
    return jsonify(result)


# 1.8 图像识别监控地点
@app.route('/identify_monitorloc', methods=['post'])
def get_monitorloc():
    data = json.loads(request.get_data().decode('utf-8'))

    try:
        pictureId = data["pictureId"]
        picname = data["screenshot"]
        assert os.path.exists(picname)

        current_location = get_time(picname, 1)
        result = {
            "code": 200,
            "message": {
                "pictureId": pictureId,
                "address": current_location
            }
        }
    except:
        result = {
            "code": 500,
            "message": "程序异常"
        }
    return jsonify(result)


# 1.9 提取图片目标特写区域（人和车）
@app.route('/target_locate/object', methods=['post'])
def locateObject():
    data = json.loads(request.get_data().decode('utf-8'))

    try:
        pictureId = data["pictureId"]
        picname = data["screenshot"]
        assert os.path.exists(picname)

        object_locs = detection_object(picname, net_detect_car_cuda1)
        plate_locs = detection_plates(picname, net_detect_plate_cuda1, device_1)
        face_locs = detection_faces(picname, net_detect_face_cuda1)
        face_person = []
        obj_mess = []

        for io in range(len(object_locs)):
            label = object_locs[io][4]
            # label 0:person/1-6:car
            
            if label == 0:
                x_left = object_locs[io][0]
                y_top = object_locs[io][1]
                x_right = object_locs[io][0] + object_locs[io][2]
                y_bottom = object_locs[io][1] + object_locs[io][3]
                
                flag = False
                for f in range(len(face_locs)):
                    center_face_x = face_locs[f][0] + face_locs[f][2] / 2
                    center_face_y = face_locs[f][1] + face_locs[f][3] / 2
                    if x_left < center_face_x < x_right and y_top < center_face_y < y_bottom:
                        face_loc = face_locs[f]
                        del face_locs[f]
                        flag = True
                        break
                if not flag:
                    face_loc = tuple((0, 0, 0, 0))

                person_dict = {"type": 1, "vehicleClassType": "", "vehicleColor": "",
                               "body": object_locs[io][:4], "face": face_loc}
                obj_mess.append(person_dict)
            else:
                car_x_left = object_locs[io][0]
                car_x_right = object_locs[io][0] + object_locs[io][2]
                car_y_left = object_locs[io][1]
                car_y_right = object_locs[io][1] + object_locs[io][3]

                flag = False
                for p in range(len(plate_locs)):
                    x = plate_locs[p][0]
                    y = plate_locs[p][1]
                    if car_x_left < x < car_x_right and car_y_left < y < car_y_right:
                        car_plate = plate_locs[p]
                        flag = True
                        # del plate_locs[p]
                        break
                if not flag:
                    car_plate = tuple((0, 0, 0, 0))
                car_dict = {"type": 2, "vehicleClassType": object_locs[io][4], "vehicleColor": object_locs[io][5],
                            "body": object_locs[io][:4], "face": car_plate}
                obj_mess.append(car_dict)

        img = cv2.imread(picname)
        for i in range(len(obj_mess)):
            cv2.rectangle(img, (obj_mess[i]['face'][0], obj_mess[i]['face'][1]),
                            (obj_mess[i]['face'][0]+obj_mess[i]['face'][2], obj_mess[i]['face'][1]+obj_mess[i]['face'][3]), (255, 0, 0), 2)
            cv2.rectangle(img, (obj_mess[i]['body'][0], obj_mess[i]['body'][1]),
                            (obj_mess[i]['body'][0]+obj_mess[i]['body'][2], obj_mess[i]['body'][1]+obj_mess[i]['body'][3]), (0, 255, 0), 2)
        cv2.imwrite('crop_object.png', img)
        # print(f'obj_mess:{obj_mess}')

        result = {
            "code": 200,
            "message": {
                "pictureId": pictureId,
                "targetList": [
                    {
                        "type": obj_mess[i]["type"],
                        "vehicleClassType": obj_mess[i]["vehicleClassType"],
                        "vehicleColor": obj_mess[i]["vehicleColor"],
                        "body": {
                            "x": obj_mess[i]["body"][0],
                            "y": obj_mess[i]["body"][1],
                            "width": obj_mess[i]["body"][2],
                            "height": obj_mess[i]["body"][3]
                        },
                        "face": {
                            "x": obj_mess[i]["face"][0],
                            "y": obj_mess[i]["face"][1],
                            "width": obj_mess[i]["face"][2],
                            "height": obj_mess[i]["face"][3]
                        }
                    }for i in range(len(obj_mess))
                ]
            }
        }
    except:
        result = {
            "code": 500,
            "message": "程序异常"
        }
    return jsonify(result)


# 初始化模型
def init_model(device):
    net = get_model('r100', fp16=True)
    net.load_state_dict(torch.load(
        'arcface_torch/glint360k_cosface_r100_fp16_0.1/backbone.pth'))
    net.eval()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    return net

if __name__ == "__main__":
    device = torch.device('cuda:0')
    # device_1 = torch.device('cuda:1')
    device_1 = device

    thread = 0.8

    net_detect_car_1_cuda1 = torch.hub.load('ultralytics/yolov5', 'yolov5l', device=device_1)
    net_detect_car_1_cuda1.eval()
    net_detect_car_1_cuda1.to(device_1)
    
    # 1.3
    from insightface.app import FaceAnalysis
    net_detect_face = FaceAnalysis(allowed_modules=['detection'])
    net_detect_face.prepare(ctx_id=0, det_thresh=thread, det_size=(640, 640))

    net_detection_persons =  torch.hub.load(
        'ultralytics/yolov5', 'yolov5l')  # or yolov5m, yolov5l, yolov5x, custom
    net_detection_persons.eval()

    net_detect_face_cuda1 = FaceAnalysis(allowed_modules=['detection'])
    net_detect_face_cuda1.prepare(ctx_id=1, det_thresh=thread, det_size=(640, 640))
    # net_detection_persons_cuda1 = torch.hub.load(
    #     'ultralytics/yolov5', 'yolov5l', device=device_1)  # or yolov5m, yolov5l, yolov5x, custom
    # # net_detection_persons_cuda1.to(device_1)
    # net_detection_persons_cuda1.eval()

    # 1.1
    net_extraction = [init_model(device), net_detect_face]

    # 1.5
    from myyolov5_plate.LPRNet.LPRNet_Test import *
    model_recog_plate = LPRNet(class_num=len(CHARS), dropout_rate=0)
    model_recog_plate.to(device)
    model_recog_plate.load_state_dict(torch.load(
        './myyolov5_plate/LPRNet/weights/Final_LPRNet_model.pth', 
        map_location=lambda storage, loc: storage))
    model_recog_plate.eval()

    model_recog_plate_cuda1 = LPRNet(class_num=len(CHARS), dropout_rate=0)
    model_recog_plate_cuda1.to(device_1)
    model_recog_plate_cuda1.load_state_dict(torch.load(
        './myyolov5_plate/LPRNet/weights/Final_LPRNet_model.pth',
        map_location=lambda storage, loc: storage))
    model_recog_plate_cuda1.eval()

    # 1.4
    from car_attribute.VehicleDC import Car_Classifier
    from plate_detector.data import cfg_mnet
    from plate_detector.models.retina import Retina
    from detect_plate import load_model as plate_load_model
    import torch.backends.cudnn as cudnn
    net_detect_car_1 = torch.hub.load('ultralytics/yolov5', 'yolov5l')
    net_detect_car_1.eval()
    net_detect_car_2 = Car_Classifier(
        num_cls=19, model_path='./car_attribute/checkpoints/epoch_39.pth',devices=device)
    # net_detect_car_2.eval()
    net_detect_car = [net_detect_car_1, net_detect_car_2]

    net_detect_car_2_cuda1 = Car_Classifier(
        num_cls=19, model_path='./car_attribute/checkpoints/epoch_39.pth', devices=device_1)
    net_detect_car_cuda1 = [net_detect_car_1_cuda1, net_detect_car_2_cuda1]

    cfg = cfg_mnet
    net_detect_plate = Retina(cfg=cfg, phase='test')
    net_detect_plate = plate_load_model(
        net_detect_plate, './plate_detector/weights/mnet_plate.pth', False)
    net_detect_plate.eval()
    cudnn.benchmark = True
    net_detect_plate = net_detect_plate.to(device)
    net_detect_plate.eval()

    net_detect_plate_cuda1 = Retina(cfg=cfg, phase='test')
    net_detect_plate_cuda1 = plate_load_model(
        net_detect_plate_cuda1, './plate_detector/weights/mnet_plate.pth', False)
    net_detect_plate_cuda1.eval()
    cudnn.benchmark = True
    net_detect_plate_cuda1 = net_detect_plate_cuda1.to(device_1)
    net_detect_plate_cuda1.eval()

    # 1.6
    import timm
    net_attribute_car = [net_detect_car_2,
                         timm.create_model('resnet50', pretrained=True).eval()]


    model_version = "model-v2"
    port = 8080
    print("service started at port {}!".format(port))
    # app.run(host='0.0.0.0', port=str(port), threaded=True)
    
    server = pywsgi.WSGIServer(('0.0.0.0', port), app)
    server.serve_forever()

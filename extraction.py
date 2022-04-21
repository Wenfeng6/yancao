from arcface_torch.inference import inference 
from detect_face import detection_faces
import cv2
import face_recognition
import numpy as np
from arcface_torch.backbones import get_model
import torch
import time

def extraction(x, net):
    features = []
    for sample in x:
        # st = time.time()
        
        face_locations = detection_faces(sample, app=net[1], flag=False)
        # print(time.time() - st)
        # print("face_locations")
        if len(face_locations) == 0:
            x, y = 0, 0
            # height, width, _  = img.shape
            is_save = 2
            face_locations = None
        else:
            x, y, width, height = face_locations[0]
            face_locations = [y, x+width, y+height, x]
            is_save = blurJudge(sample, [x, y, width, height])
            if is_save == 0:
                image = face_recognition.load_image_file(sample)
                # face_landmarks_list = face_recognition.face_landmarks(image, model='small')
                face_landmarks_list = face_recognition.face_locations(image, model='hog')
                if len(face_landmarks_list) == 0:
                    is_save = 1
        print(is_save)

        # print(face_locations)
        # img = cv2.imread(sample)
        # tmp = img[face_locations[0]:face_locations[2], face_locations[3]:face_locations[1]]
        # savename = sample.split('.')[0] + '_crop.png'
        # cv2.imwrite(savename, tmp)
        # print(face_locations)
        feature = inference(sample, face_locations, net[0])
        features.append(feature)


    return features, is_save


def blurJudge(picname, location):
    [x, y, width, height] = location
    img = cv2.imread(picname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img.shape)
    img = img[y:y+height, x:x+width]
    img = cv2.resize(img, (112, 112))

    imageVar = cv2.Laplacian(img, cv2.CV_64F).var()

    # x, y = img.shape
    # D = 0
    # for i in range(x-2):
    #     for j in range(y-2):
    #         D += (img[i+2, j] - img[i, j])**2
    # # 大于100 图像清晰 返回0
    return 0 if imageVar > 100 else 1

def init_model():
    net = get_model('r100', fp16=True)
    net.load_state_dict(torch.load(
        'arcface_torch/glint360k_cosface_r100_fp16_0.1/backbone.pth'))
    net.eval()
    net.cuda()
    return net


if __name__ == '__main__':
    from insightface.app import FaceAnalysis
    net_detect_face = FaceAnalysis(allowed_modules=['detection'])
    net_detect_face.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    net_detection_persons =  torch.hub.load(
        'ultralytics/yolov5', 'yolov5l')  # or yolov5m, yolov5l, yolov5x, custom
    net_detection_persons.eval()

    # 1.1
    net_extraction = [init_model(), net_detect_face]
    
    feature, is_save = extraction(['test1201/新建文件夹/2021-12-01-09-13-50-11.png'], net_extraction)
    feature = feature[0]
    str_feature = ""
    for index in range(feature.shape[1]):
        str_feature += str(feature[0, index]) + " "
    # print(str_feature)
    # print(is_save)

    # piclist = ['samples/1.png','samples/2.png','samples/3.png','samples/testfilter1.jpg','samples/testfilter2.jpg','samples/testfilter3.jpg']
    # score = []
    # for i in piclist:
    #     score.append(blurJudge(i))
    # print(score)

import cv2
import torch
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from torch.cuda.amp import autocast as autocast

def detection_faces(pic, app, flag=True):
    result = []
    
    # app = FaceAnalysis()
    # app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
    # # app.det_model.cuda()
    if isinstance(pic, str):
        img = np.array(cv2.imread(pic))
    else:
        img = np.array(pic)
    # with torch.no_grad():
    # with autocast():
    faces = app.get(img)

    for i in range(len(faces)):
        # print(faces[i].bbox)

        x = int(faces[i].bbox[0])
        y = int(faces[i].bbox[1])
        width = int(faces[i].bbox[2] - faces[i].bbox[0])
        height = int(faces[i].bbox[3] - faces[i].bbox[1])

        if flag:
            x = max(0, x - width // 2)
            y = max(0, y - height // 2)
            width = width * 2
            height = height * 2
            if x + width > img.shape[1]:
                width = img.shape[1] - x
            if y + height > img.shape[0]:
                height = img.shape[0] - y
            faceloc = (x, y, width, height)
        else:
            if x < 0:
                width += x
                x = 0
            if y < 0:
                height += y
                y = 0
            if x + width > img.shape[1]:
                width = img.shape[1] - x
            if y + height > img.shape[0]:
                height = img.shape[0] - y
            faceloc = (x, y, width, height)
        result.append(faceloc)
    

    return result # top right bottom left

    # face = Image.fromarray(face)
    # image = Image.fromarray(image)
    # draw = ImageDraw.Draw(image)
    # draw.line([ (location[3], location[0]),
    #             (location[1], location[0]),
    #             (location[1], location[2]),
    #             (location[3], location[2]),
    #             (location[3], location[0])],width=2, fill='red')
    # face.save("{}.png".format(index))
    # image.save("location.png")
if __name__ == "__main__":
    from insightface.app import FaceAnalysis
    net_detect_face = FaceAnalysis(allowed_modules=['detection'])
    net_detect_face.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    pic = "samples/testperson4.jpg"
    face_locs = detection_faces(pic, net_detect_face)
        
    import cv2
    img = cv2.imread(pic)
    for i in range(len(face_locs)):
        cv2.rectangle(img, (face_locs[i][0], face_locs[i][1]), \
        (face_locs[i][0]+face_locs[i][2], face_locs[i][1]+face_locs[i][3]), (0,0,255), 2)
    cv2.imwrite('crop_persons.png', img)
    
    # app = FaceAnalysis()
    # app.prepare(ctx_id=0, det_size=(640, 640))
    # img = np.array(cv2.imread(pic))
    # faces = app.get(img)
    # for i in range(len(faces)):
    #     print(faces[i].bbox)
    # rimg = app.draw_on(img, faces)
    # cv2.imwrite("./t1_output.jpg", rimg)
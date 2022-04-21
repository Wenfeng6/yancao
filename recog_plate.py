import numpy as np
from myyolov5_plate.LPRNet.LPRNet_Test import *
from recog_plate_ocr import get_platemess
# from myyolov5_plate.MTCNN import *


def get_license(picname, lprnet, ocr):
    if isinstance(picname, str):
        img = cv2.imread(picname)
    else:
        img = picname
    # device = '0'
    # cuda = torch.cuda.is_available()
    # device = torch.device('cuda:0' if cuda else 'cpu')
    # lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    # lprnet.to(device)
    # lprnet.load_state_dict(torch.load(
    #     './myyolov5_plate/LPRNet/weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))

    # txt=plot_one_box_plate_direct(img, lprnet)
    color = check_plate_color(img)
    im = cv2.resize(img, (94, 24), interpolation=cv2.INTER_CUBIC)
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5)*0.0078125
    data = torch.from_numpy(im).float().unsqueeze(
        0).to('cuda:0')  # torch.Size([1, 3, 24, 94])
    #data = STN(data)
    # a = time.time()
    with torch.no_grad():
        preds = lprnet(data)
    # b = time.time()
    # print(b-a)

    preds = preds.cpu().numpy()  # (1, 68, 18)
    
    labels, pred_labels = decode(preds, CHARS)
    
    # font = cv2.FONT_HERSHEY_SIMPLEX
    txtt = u"%s" % (labels[0])
    txt = txtt.encode('utf-8').decode('utf-8')
    txt = "".join(txt.split())

    chn_list = ['京', '沪', '津', '渝', '鲁', '冀', '晋', '蒙', '辽',
                '吉', '黑', '苏', '浙', '皖', '闽', '赣', '港',
                '豫', '湘', '鄂', '粤', '桂', '琼', '川', '澳',
                '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '台']
    
    # print("plate license ans:")
    # print(txt)
    def judge_plate_legal(platetxt):
        for c in platetxt:
            if not ('0' <= c <= '9' or 'a' <= c <= 'z' or 'A' <= c <= 'Z'):
                return False
        return True
    
    if not (9 > len(txt) > 6 and judge_plate_legal(txt)):
        txt_c, txt_e = get_platemess(picname, ocr)
        # txt = "未知"
        if len(txt_e) == 0:
            txt = ''
        elif len(txt_c) == 0:
            if txt[0] in chn_list:
                txt = txt[0] + txt_e
        else:
            txt = txt_c + txt_e
        plateReli = "0"
        platecharReli = ""
    else:
        plateReli, platecharReli = get_reliability(txt, pred_labels)
    # print(f'recog ans:{txt}')
    return txt, color, plateReli, platecharReli

def get_reliability(txt, pred_labels):
    if len(pred_labels) == 0:
        return "0", ""
    preds = np.zeros([len(txt)])
    for i in range(len(txt)):
        alpha = 0.3 if (i == 0 and txt[0] == '闽') else 0.4
        preds[i] = alpha * pred_labels[0][i] + (1-alpha) * 100
    pred_max = np.random.randint(90, 96)
    pred_min = np.random.randint(55, 65)
    preds = np.clip(preds, pred_min, pred_max)
    preds = np.ceil(preds).astype(np.uint8)
    # print(preds)
    
    platecharReli = txt[0] + '-' + str(preds[0])
    for i in range(1, len(txt)):
        platecharReli += ", {}-{}".format(txt[i], preds[i])
    plateReli = str(int(np.mean(preds)))
    return plateReli, platecharReli


def check_plate_color(img):

    # img = img[25:35, 30:60, :]
    # cv2.imwrite('./part.png', img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # print(hsv)

    area = img.shape[0] * img.shape[1]

    lower_blue = np.array([100., 43., 46.])
    upper_blue = np.array([124., 255., 255.])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_cnt = cv2.countNonZero(mask_blue) / area
    print(blue_cnt)

    lower_green = np.array([55., 30., 100.])
    upper_green = np.array([120., 160., 255.])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    green_cnt = cv2.countNonZero(mask_green) / area
    print(green_cnt)

    lower_yellow = np.array([26., 43., 46.])
    upper_yellow = np.array([34., 255., 255.])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_cnt = cv2.countNonZero(mask_yellow) / area
    print(yellow_cnt)

    lower_black = np.array([0., 0., 0.])
    upper_black = np.array([50., 50., 50.])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    black_cnt = cv2.countNonZero(mask_black) / area
    print(black_cnt)

    lower_white = np.array([0., 0., 120.])
    upper_white = np.array([180., 50., 255.])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    white_cnt = cv2.countNonZero(mask_white) / area
    print(white_cnt)

    mask = mask_green > 0
    mask = mask.astype(np.uint8)
    # cv2.imwrite('./test.png', mask.reshape(img.shape[0], img.shape[1], 1) * img)

    if blue_cnt > 0.3:
        return '蓝色'
    elif yellow_cnt > 0.3:
        return '黄色'
    elif green_cnt > 0.2:
        return '绿色'
    elif black_cnt > 0.3:
        return '黑色'
    elif white_cnt > 0.3:
        return '白色'
    else:
        return ''

if __name__ == "__main__":
    device = torch.device('cuda:0')
    from myyolov5_plate.LPRNet.LPRNet_Test import *
    model_recog_plate = LPRNet(class_num=len(CHARS), dropout_rate=0)
    model_recog_plate.to(device)
    model_recog_plate.load_state_dict(torch.load(
        './myyolov5_plate/LPRNet/weights/Final_LPRNet_model.pth', 
        map_location=lambda storage, loc: storage))
    model_recog_plate.eval()

    picname = "/home/nqiv_admin/ycnqiv/plate/4.jpeg"
    license, color, plateReli, platecharReli = get_license(picname, model_recog_plate)
    print(license, color, plateReli, platecharReli)
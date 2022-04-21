from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import tesserocr


def get_firstchar(img):
    chn_list = ['京', '沪', '津', '渝', '鲁', '冀', '晋', '蒙', '辽', 
                '吉', '黑', '苏', '浙', '皖', '闽', '赣', '港',
                '豫', '湘', '鄂', '粤', '桂', '琼', '川', '澳',
            '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '台']
    correct_dict = {'淅': '浙'}
    txt_c = tesserocr.image_to_text(img, lang='chi_sim')
    print(f'txt_c:{txt_c}')
    char_loc = ''
    char_second = ''
    if len(txt_c) != 0:
        # print(f'plate firstchar ocr ans:{txt_c[0]}')
        if txt_c[0] in correct_dict.keys():
            char_first = correct_dict[txt_c[0]]
        else:
            char_first = txt_c[0]
        if char_first in chn_list:
            char_loc = char_first
        
        if len(txt_c) > 1 and 'A' <= txt_c[1] <= 'Z':
            char_second = txt_c[1]
        
    return char_loc, char_second

def get_numberchar(img, char_sec):
    blur_list = ['c', 'k', 'o', 's', 'u', 'v', 'w', 'x', 'z']
    space_list = [' ', '-', '.', '\n']
    filter_list = [str(i) for i in range(10)]
    char_list = [chr(i) for i in range(65,91)]
    filter_list.extend(char_list)
    filter_list.extend(space_list)
    filter_list.extend(blur_list)

    txt_e = tesserocr.image_to_text(img)
    print(f'txt_e:{txt_e}')
    if len(txt_e) != 0:
        while len(txt_e) != 0 and txt_e[-1] in space_list:
            txt_e = txt_e[:-1]
        
        filter_text0 = ''.join((filter(lambda val: val in filter_list, txt_e)))
        filter_text = ''.join((map(lambda val: val.upper(), filter_text0)))
        ind_space = -1
        print(f'filter_text:{filter_text}')
        for i in range(len(space_list) - 1):
            ind_space_tmp = filter_text.find(space_list[i])
            ind_space = max(ind_space, ind_space_tmp)
        # print(f'ind_space:{ind_space}')
        if ind_space in [-1]:
            return ''
        if (len(filter_text) - ind_space) not in [6, 7]:
            return ''
        if len(char_sec) != 0:
            char_second = char_sec
        else:
            ind_tmp = ind_space
            flag = False
            while ind_tmp != 0:
                char_second = filter_text[ind_tmp - 1]
                if 'A' <= char_second <= 'Z':
                    flag = True
                    break
                ind_tmp -= 1
            if not flag:
                return ''
        char_num = char_second + filter_text[ind_space+1:]
    else:
        char_num = ''
    return char_num

def get_platemess(picname, ocr):
    if isinstance(picname, str):
        img = Image.open(picname)
        img_ocr = cv2.imread(picname)
    else:
        img = Image.fromarray(picname)
    width, height = img.size
    # 车牌高度在50~60效果较好
    if height > 60 or height < 50:
        rate = width / height
        height = 52
        width = int(height * rate)
        img = img.resize((width, height))
    pad_size = 3
    img = ImageOps.expand(img, border=(pad_size, pad_size, pad_size, pad_size), fill=0)
    result = ocr(img_ocr)[0][1][0]
    if len(result) == 8 or len(result) == 7:
        return result[0], result[1:]
    char_loc, char_second = get_firstchar(img)
    char_num = get_numberchar(img, char_second)
    
    return char_loc , char_num
    


if __name__ == "__main__":
    # pic = "samples/testcolor2.jpg"
    # img = Image.open(pic)
    # npimg = np.array(img)
    # H, W, _ = npimg.shape
    # npimg = npimg[:, :int(W/3)]
    # cv2.imwrite('testplate.png', npimg)
    # cropimg = Image.fromarray(npimg)

    # txt_c = tesserocr.image_to_text(img, lang='chi_sim')
    # txt_e = tesserocr.image_to_text(img)
    # print(txt_c)
    # print(txt_e)

    pic = "samples/crop_plate2.png"

    from agentocr import OCRSystem
    from agentclpr.infer.utility import clp_det_model, det_model, cls_model, rec_model, rec_char_dict_path, base64_to_cv2

    ocr = OCRSystem(det_model=det_model,
                    cls_model=cls_model,
                    rec_model=rec_model,
                    rec_char_dict_path=rec_char_dict_path,
                    det_db_score_mode='slow',
                    det_db_unclip_ratio=1.3,
                    )

    print(get_platemess(pic, ocr))
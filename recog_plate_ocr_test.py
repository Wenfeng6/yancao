from agentocr import OCRSystem
from agentclpr.infer.utility import clp_det_model, det_model, cls_model, rec_model, rec_char_dict_path, base64_to_cv2

from PIL import Image
import cv2

print(rec_char_dict_path)

ocr = OCRSystem(det_model=det_model,
                cls_model=cls_model,
                rec_model=rec_model,
                rec_char_dict_path=rec_char_dict_path,
                det_db_score_mode='slow',
                det_db_unclip_ratio=1.3,
                )


img = cv2.imread('plate/3.png')

print(ocr(img))
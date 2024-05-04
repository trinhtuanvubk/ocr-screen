import cv2
from paddleocr import PaddleOCR, draw_ocr


def paddle_ocr_init():
    ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_thresh=0.7) # need to run only once to download and load model into memory
    return ocr

def paddle_ocr_pl(image, ocr_reader):
    result = ocr_reader.ocr(image, cls=False)
    result = result[0]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    return boxes, txts
    
    
    
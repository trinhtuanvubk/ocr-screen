#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Part of this implementation is borrowed from https://modelscope.cn/studios/damo/cv_ocr-text-spotting/file/view/master/app.py

import sys
import numpy as np

from scenario.damo_ocr.text_detection import TextDetection
from scenario.damo_ocr.text_recognition import TextRecognition

from scenario.damo_ocr.util import general_text_reading_visualization


class GeneralTextReading(object):
    """
    Description:
      class definition of GeneralTextReading pipeline: 
      (1) algorithm interfaces for general text reading (detection + recognition)
      (2) document layout and structure are not taken into consideration

    Caution:
    """

    def __init__(self, configs):
        """
        Description:
          initialize the class instance
        """

        # initialize and launch pipiline
        self.text_detection_module = TextDetection(configs['text_detection_configs'])
        self.text_recognition_module = TextRecognition(configs['text_recognition_configs'])

    def __call__(self, image):
        """
        Description:
          detect and recognize all text instances (those virtually machine-identifiable) from the input image

        Parameters:
          image: the image to be processed, assume that it is a *full* image potentially containing text instances

        Return:
          final_result: text reading result
        """

        # initialize
        final_result = []
        det_result = None
        rec_result = None

        # perform text detection and recognition successively
        if image is not None:
            det_result = self.text_detection_module(image)
            rec_result = self.text_recognition_module(image, det_result)    

        # assembling
        for i in range(det_result.shape[0]):
            item = {}
            item['position'] = det_result[i].tolist()
            item['content'] = rec_result[i]['text']
            final_result.append(item)

        return final_result

    def release(self):
        """
        Description:
          release all the resources
        """

        if self.text_detection_module is not None:
            self.text_detection_module.release()
        
        if self.text_recognition_module is not None:
            self.text_recognition_module.release()

        return 

def damo_ocr_init():
    # configure
    configs = dict()
    
    text_detection_configs = dict()
    text_detection_configs['from_modelscope_flag'] = True
    # text_detection_configs['model_path'] = 'damo/cv_resnet18_ocr-detection-line-level_damo'
    text_detection_configs['model_path'] = 'damo/cv_resnet18_ocr-detection-db-line-level_damo'
    configs['text_detection_configs'] = text_detection_configs

    text_recognition_configs = dict()
    text_recognition_configs['from_modelscope_flag'] = True
    text_recognition_configs['lang'] = 'en'
    text_recognition_configs['model_path'] = 'damo/cv_convnextTiny_ocr-recognition-general_damo'  # alternatives: 'damo/cv_convnextTiny_ocr-recognition-scene_damo', 'damo/cv_convnextTiny_ocr-recognition-document_damo', 'damo/cv_convnextTiny_ocr-recognition-handwritten_damo' 
    # text_recognition_configs['model_path'] = 'modelscope/hub/damo/cv_convnextTiny_ocr-recognition-general_damo'
    configs['text_recognition_configs'] = text_recognition_configs

    # initialize
    text_reader = GeneralTextReading(configs)
    
    return text_reader

def damo_ocr_pl(image, text_reader, visual=False):
    # run
    final_result = text_reader(image)

    if True:
        print (final_result)
    if visual==True:
        # visualize
        output_image = general_text_reading_visualization(final_result, image)

        # release
        text_reader.release()

        return final_result, output_image
    else:
        boxes = [i['position'] for i in final_result]
        txts = [i['content'] for i in final_result]
        
        return boxes, txts
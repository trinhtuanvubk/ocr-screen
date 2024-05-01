#  python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

# pip install "paddleocr>=2.0.1" # Recommend to use version 2.0.1+from paddleocr import PaddleOCR,draw_ocr


from paddleocr import PaddleOCR,draw_ocr

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = '../frames_test/200.jpg'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
print("==========")
print(txts)
print("==========")
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='../fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('../results/paddle/200.jpg')
# /Users/vutrinh/.cache/torch/hub/checkpoints/dbnet_resnet50-oclip_1200e_icdar2015_20221102_115917-bde8c87a.pth

from mmocr.apis import MMOCRInferencer
# Load models into memory
ocr = MMOCRInferencer(det='DBNet', rec='SAR')
# Perform inference
ocr('frames/200.jpg', out_dir="results/mmocr", save_vis=True)
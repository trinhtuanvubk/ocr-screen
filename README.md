# ocr-screen

# Install
- To install common packages:
```bash
pip install -r requirements.txt
```

# Methods

### PaddleOCR
- To install:
```bash
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install "paddleocr>=2.0.1
```

- Run a test:
```bash
cd test
python3 paddleocr_test.py
```
<!-- Remove MMOCR pipeline because I can not install old and silly packages -->

### DamoOCR
- Install: just need to install modelscope to download model checkpoint
- Test: There are no tests right now
### Pipeline
- To run pipeline:
```bash
python3 pipeline.py \
--recording_path sample.webm \
--method paddle_ocr \
--step 5
```

Where:

    -`method`: currently supports `paddle_ocr` or `damo_ocr`
    -`step`: is frame step 

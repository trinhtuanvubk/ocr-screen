# ocr-screen

# Methods

### PaddleOCR
- To install:
```bash
python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install "paddleocr>=2.0.1
```

- Run a test:
```bash
python3 paddleocr_test.py
```

### MMOCR
- To install:
```bash
pip install openmim
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
mim install -e .
```
NOTE: this method is not available

### Pipeline
- To run pipeline:
```bash
python3 pipeline.py \
--recording_path sample.webm \
--method paddleocr \
--step 5
```

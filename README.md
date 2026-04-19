# ASL Recognition (Hackathon)

Real-time prototype:

`webcam -> MediaPipe hand landmarks -> feature vector -> classifier -> on-screen prediction`

## Folder layout

- `data/asl_data.csv` - collected features
- `models/asl_model.h5` - trained model (stored via `joblib`)
- `models/label_encoder.pkl` - label encoder for mapping class index -> label
- `src/hand_tracking.py` - MediaPipe hand tracking
- `src/collect_data.py`- webcam data collection tool
- `src/train_model.py` - trains a simple classifier
- `src/predict_realtime.py` - live prediction + smoothing
- `utils/preprocess.py` - feature normalization
- `utils/visualization.py` - OpenCV drawing helpers
- `camera_test.py` - sanity check for webcam
- `main.py` - runs real-time prediction

## Install

```bash
pip install -r requirements.txt
```

## 1) Test webcam

```bash
python camera_test.py
```

## 2) Collect data

```bash
python src/collect_data.py
```

Controls:
- press a mapped key to set label (A-Z mapped to `a-z`, phrases mapped to `1-0`)
- press `SPACE` to save a sample
- press `q` to quit

## 3) Train model

```bash
python src/train_model.py
```

## 4) Predict in real time

```bash
python main.py
```

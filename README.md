# Rock-Paper-Scissors Hand Detection with YOLOv8

A machine learning project that recognizes **rock, paper, or scissors** hand gestures, both from images and **live from your webcam**. It fine-tunes a small **YOLOv8** image classifier on a public rock-paper-scissors dataset.

This was an assignment for a Machine Learning course. (*Batu, gunting, kertas* is Indonesian for rock, scissors, paper.)

## How it works

1. **Prepare the data**: the notebook sorts the Roboflow dataset into one folder per class (`batu` = rock, `gunting` = scissors, `kertas` = paper).
2. **Train**: start from the pre-trained `yolov8n-cls.pt` model and train for 15 epochs on 224×224 images.
3. **Detect live** with `cek_kamera.py`:
   - It finds your hand in the webcam image using a **skin-color mask** (HSV color range).
   - It crops a square around the largest hand-shaped area.
   - The trained model classifies the crop, and the label and confidence are drawn on screen: green when the model is more than 70% sure, red otherwise.
   - Press `q` to quit.

## Tech stack

Python, Ultralytics YOLOv8, OpenCV, NumPy, pandas, Jupyter

## Getting started

You need Python 3.9+ and a webcam.

```bash
pip install ultralytics opencv-python numpy pandas notebook
```

1. Open `Rock_Paper_Scissors_YOLO_Detection.ipynb` and run the cells to prepare the data and train the model.
2. Run the live demo:

   ```bash
   python cek_kamera.py
   ```

**Before running**, update the file paths: the notebook and `cek_kamera.py` still use the Windows paths from the original computer (for example `C:\runs\classify\rps_final_success\weights\best.pt`). Point them to your dataset folder and to the `best.pt` file that training creates.

## Project structure

| Path | What it is |
| --- | --- |
| `Rock_Paper_Scissors_YOLO_Detection.ipynb` | Data preparation, training, and testing |
| `cek_kamera.py` | Live webcam detection |
| `yolov8n-cls.pt` | Pre-trained YOLOv8 nano classifier (starting point) |
| `batu-gunting-kertas-2/` | Dataset split into `train`, `valid`, and `test` |

## Dataset

[batu-gunting-kertas](https://universe.roboflow.com/agedetection/batu-gunting-kertas-zlhyg) from Roboflow Universe, licensed CC BY 4.0.

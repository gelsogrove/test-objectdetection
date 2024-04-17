#!/bin/bash

# Creazione e attivazione dell'ambiente virtuale
python3 -m venv yolo_env
source yolo_env/bin/activate

# Clonazione del repository di YOLOv5
git clone https://github.com/ultralytics/yolov5.git


# Installazione di pip
python3 -m ensurepip --default-pip

# Reinstalla o aggiorna setuptools
 
python3 -m pip install --upgrade --force-reinstall setuptools

# Aggiornamento di pip
python3 -m pip install -U pip

python3 -m pip install onnx

python3 -m  pip install --upgrade onnxscript

python3 -m  pip install onnxruntime

python3 -m pip install opencv-python

python3 -m pip  install matplotlib
 




# Installazione delle dipendenze da requirements.txt
python3 -m pip install -r yolov5/requirements.txt

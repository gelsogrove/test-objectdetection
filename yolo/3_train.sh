#!/bin/bash

# Percorsi dei file di configurazione
DATA_PATH="data/Cripta1-1/data.yaml"
CFG_PATH="models/yolov5s.yaml"
WEIGHTS_PATH="yolov5s-seg.pt"

# Impostazioni per l'addestramento
IMG_SIZE=640
BATCH_SIZE=64
EPOCHS=100

# Cambia directory
cd yolov5

# Esegui lo script di addestramento con le variabili definite
python3 train.py --img $IMG_SIZE --batch $BATCH_SIZE --epochs $EPOCHS --data $DATA_PATH --cfg $CFG_PATH --weights $WEIGHTS_PATH    

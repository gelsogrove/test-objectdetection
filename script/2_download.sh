#!/bin/bash

# Change directory to data/images
cd "yolov5/data"

# Python code
python_code=$(cat << 'EOF'
import torch
print('torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

from roboflow import Roboflow

def download_dataset():
    try:
        rf = Roboflow(api_key="yOkpFuVLsae0KQ9qLG1h")
        project = rf.workspace("test01-vyobk").project("cripta1")
        version = project.version(1)
        dataset = version.download("yolov5")
        return dataset
    except Exception as e:
        print(f"Errore on download: {e}")
        exit(1)

download_dataset()
EOF
)

# Execute Python code
echo "$python_code" > temp_script.py
python3 temp_script.py
rm temp_script.py


 

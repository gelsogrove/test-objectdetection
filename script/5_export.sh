python3 yolov5/export.py --weights yolov5/runs/train/exp/weights/best.pt \
                         --img 640 480 \
                         --include onnx \
                         --simplify \
                         --dynamic \
                         --opset 14

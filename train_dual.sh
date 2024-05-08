python  train_dual.py \
    --data data/taco.yaml \
    --img 640 \
    --batch 24 \
    --device 0 \
    --weights /home/vladimir/Work/Projects/yolo9/models/yolov9-c.pt \
    --workers 8 \
    --cfg models/detect/yolov9-c.yaml \
    --name yolov9-c-taco-freeze-bn-ft-nwd \
    --hyp hyp.scratch-high.yaml \
    --min-items 0 \
    --epochs 500 \
    --close-mosaic 15 \
    --freeze 10 \

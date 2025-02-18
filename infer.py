import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="YOLOv8 inference script")
parser.add_argument(
    "--model",
    type=str,
    default="runs/detect/train/yolov8s_100epochs/weights/best.pt",
    help="Path to YOLO weights (default: runs/detect/train/yolov8s_100epochs/weights/best.pt)"
)
parser.add_argument(
    "--source",
    type=str,
    default="test/images",
    help="Path to data for inference (default: test/images folder in project directory)"
)
parser.add_argument(
    "--save",
    action="store_true",
    help="Save predictions"
)

if __name__ == "__main__":
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.predict(source=args.source, save=args.save)

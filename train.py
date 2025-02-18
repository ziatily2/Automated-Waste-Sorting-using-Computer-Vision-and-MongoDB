import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="YOLOv8 training and validation script")
parser.add_argument(
    "--model_name",
    default="yolov8s",
    help="Name of the YOLO model to use (e.g., yolov8n, yolov8s, yolov8m, etc.)"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs to train the model"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="custom_data.yaml",
    help="Path to your YOLO dataset YAML file"
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Load the YOLO model (e.g., yolov8s.pt)
    model = YOLO(args.model_name + ".pt")

    # Train the model
    model.train(
        data=args.data_path,
        epochs=args.epochs,
        patience=25,
        imgsz=640,
        device=0,  # 0 = first GPU, 'cpu' = CPU only, or another GPU index
        name=f"{args.model_name}_{args.epochs}epochs",
        pretrained=True,
        optimizer="SGD",
    )

    # Validate the model after training
    metrics = model.val()
    print("Validation metrics:", metrics)

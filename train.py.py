from ultralytics import YOLO

def main():
    # Load pretrained YOLOv11 model
    model = YOLO("yolov11n.pt")

    # Train the model
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="military_detection"
    )

    print("Training completed successfully!")

if __name__ == "__main__":
    main()

from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to image/video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--weights", type=str, 
                        default="runs/detect/military_detection/weights/best.pt",
                        help="Path to trained weights")
    args = parser.parse_args()

    # Load trained model
    model = YOLO(args.weights)

    # Run prediction
    model.predict(
        source=args.source,
        conf=args.conf,
        save=True
    )

    print("Detection completed successfully!")

if __name__ == "__main__":
    main()

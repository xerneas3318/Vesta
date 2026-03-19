from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
PT_PATH = BASE_DIR / "yolo26s.pt"


def main() -> None:
    if not PT_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {PT_PATH}")

    model = YOLO(str(PT_PATH))
    onnx_path = model.export(format="onnx", opset=12, simplify=True, dynamic=True)
    print(f"ONNX export complete: {onnx_path}")


if __name__ == "__main__":
    main()

from ultralytics import YOLO

model = YOLO("yolo11l.pt")

model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=1280,
    device="mps",       # Mac'te MPS. Baska bilgisayardaysan "cpu" veya "0" (CUDA) yap
    project="runs/detect",
    name="drone_detect",
    exist_ok=True
)

print("Egitim tamamlandi!")
print("best.pt: runs/detect/drone_detect/weights/best.pt")

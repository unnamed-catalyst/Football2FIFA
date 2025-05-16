from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict(f"input_videos/DFL Bundesliga Data Shootout/train/A1606b0e6_0/A1606b0e6_0 (1).mp4", save=True)

print(results[0])
print("**************")
for box in results[0].boxes:
    print(box)


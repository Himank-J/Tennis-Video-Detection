from ultralytics import YOLO 

model = YOLO('models/best.pt')

result = model.track('input_data/input_video.mp4',conf=0.2, save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)
    print('---------------')
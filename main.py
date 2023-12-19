from ultralytics import YOLO
import cv2
from PIL import Image

model = YOLO('yolov8m-pose.pt')
source = './list.streams'

results = model(source, stream=True)

for result in results:
    annotated_image = result.plot()
    cv2.imshow(f'{result.path}', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

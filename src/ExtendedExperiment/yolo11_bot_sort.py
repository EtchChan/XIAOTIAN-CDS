from ultralytics import YOLO
import os
import cv2

model = YOLO("weights/best.pt")


# List of image paths
root = "VisDrone2019-MOT-test-dev/sequences/uav0000370_00001_v"
image_paths = os.listdir(root)
sorted_files = sorted(image_paths, key=lambda x: int(x.replace('img', '').split('.')[0]))

first_image = cv2.imread(os.path.join(root, sorted_files[0]))
# Initialize tracking
model.track(first_image,persist=True)  # initialize persistent tracking
# Process each image sequentially
for image in sorted_files:
    path = os.path.join(root, image)
    # Read image
    img = cv2.imread(path)
    
    # Run tracking on the image
    result = model.track(img, persist=True)  # track and maintain IDs
    
    # Plot and save results
    out = result[0].plot()
    cv2.imwrite(f"results/mot2/{image}", out)

# Reset tracking when done
model.track(persist=False)  # reset tracking


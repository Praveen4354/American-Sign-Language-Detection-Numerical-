import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)


offset = 20
imgSize = 300

# Folder to store images
base_folder = "data"
if not os.path.exists(base_folder):
    os.makedirs(base_folder)

# Counter to track images per alphabet
counter = 0
current_alphabet ='done'

start_time = time.time()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Check if hand coordinates fall within the frame
        if x - offset > 0 and y - offset > 0 and x + w + offset < img.shape[1] and y + h + offset < img.shape[0]:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

            key = cv2.waitKey(1)

            if key == ord("a"):
                counter += 1
                folder = os.path.join(base_folder, f"{current_alphabet}")
                if not os.path.exists(folder):
                    os.makedirs(folder)
                file_name = f'{current_alphabet}_{counter:02d}.jpg'
                file_path = os.path.join(folder, file_name)
                cv2.imwrite(file_path, imgWhite)
                print(f"Saved: {file_path}")

                # Check if 150 images captured for the current alphabet
                if counter >= 150:
                    counter = 0
                    current_alphabet = chr(ord(current_alphabet) + 1)
                    elapsed_time = time.time() - start_time
                    print(f"Time taken for {current_alphabet}: {elapsed_time:.2f} seconds")
                    if current_alphabet > 'Z':
                        break
                    time.sleep(10)  # Wait for 10 seconds before moving to the next alphabet

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

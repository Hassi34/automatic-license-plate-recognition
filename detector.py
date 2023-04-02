from ultralytics import YOLO
import cv2
import cvzone
import math
from utils.common import read_yaml, get_timestamp
from utils.ocr import getNumberPlateText
from utils.addDataToDatabase import uploadNumberPlateImage, updateRecordInDB
from paddleocr import PaddleOCR
import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

def main(config_path):

    config = read_yaml(config_path)
    images_dir = config['IMAGES_DIR']
    detector_path = config['DETECTOR_MODEL_PATH']
    video_path = config['VIDEO_PATH']
    mask_path = config['MASK_PATH']
    target_vehicles = config['TARGET_VEHICLES']
    pointLocation = config['POINT_LOCATION']

    ocrModel = PaddleOCR(lang = 'en')

    cap = cv2.VideoCapture(video_path)
    cap.set(3, 1280)
    cap.set(4, 720)
    model = YOLO(detector_path)
    mask = cv2.imread(mask_path)

    while True:
        success, frame = cap.read()
        imgRegion = cv2.bitwise_and(frame, mask)
        results = model(source = imgRegion, show = False, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
                w, h = x2 - x1, y2 - y1
                bbox = (x1, y1, w, h)
                # Confidence
                conf = math.ceil((box.conf[0]*100))/100
                # Class Name
                cls = int(box.cls[0])
                currentClass = model.names[cls]

                if currentClass in ['vehicle', 'license-plate'] and conf > 0.3:
                    numberPlateText = None
                    if currentClass == 'license-plate':
                        croppedImg = frame[y1:y1+h, x1:x1+w]
                        uniqueImgName = get_timestamp("captured")
                        numberPlateImgPath = os.path.join(images_dir, uniqueImgName+".png")

                        cv2.imwrite(numberPlateImgPath, croppedImg)
                        
                        numberPlateText = getNumberPlateText(ocrModel, numberPlateImgPath)
                        
                        if numberPlateText:
                            if numberPlateText in target_vehicles:
                                cvzone.cornerRect(frame, bbox, colorR=(0, 0, 255))
                                cvzone.putTextRect(frame, f'Suspect Detected : {numberPlateText}', 
                                                (max(0, x1), max(35, y1)),
                                                    colorR=(0, 0, 255), scale=2, thickness=2)
                                cv2.putText(frame, f'Suspect Detected : {numberPlateText}',
                                             (95, 92), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
                                cv2.rectangle(frame, (0, 0), (1000, 120), (0,0,0), -1)
                                cv2.putText(frame, f'Suspect Detected : {numberPlateText}',
                                                                (15, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                                cv2.putText(frame, f'Location : {pointLocation}',
                                                        (15, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                                updateRecordInDB(numberPlateText, location=pointLocation)
                                uploadNumberPlateImage(numberPlateImgPath)
                                os.remove(numberPlateImgPath)
                            else:
                                cvzone.cornerRect(frame, bbox, colorR=(255, 0, 255))
                                cvzone.putTextRect(frame, f'Number plate : {numberPlateText}', 
                                                (max(0, x1), max(35, y1)),
                                                    colorR=(255, 0, 255), scale=2, thickness=2)
                                os.remove(numberPlateImgPath)
                        else:
                            os.remove(numberPlateImgPath)
                            cvzone.cornerRect(frame, bbox, colorR=(255, 0, 255))
                            cvzone.putTextRect(frame, "Number Plate", 
                                        (max(0, x1), max(35, y1)),
                                            colorR=(255, 0, 255), scale=2, thickness=2)
                    else:
                        cvzone.cornerRect(frame, bbox, colorR=(255, 0, 255))
                        cvzone.putTextRect(frame, 'Vehicle',
                                          (max(0, x1), max(35, y1)), scale=2, thickness=2)
                
            cv2.imshow("Image", frame)
            cv2.waitKey(1)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "configs/config.yaml")
    parsedArgs = args.parse_args()
    main(parsedArgs.config)
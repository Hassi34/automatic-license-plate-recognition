import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

def getNumberPlateText(ocr_model, img_path):
    detections = ocr_model.ocr(img_path)
    for detection in detections[0]:
        text = detection[1][0]
        if len(text) ==8: return text
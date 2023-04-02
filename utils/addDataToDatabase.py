import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime
import os 

from dotenv import load_dotenv
load_dotenv()

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    "databaseURL": os.environ['DATABASE_URL'],
    "storageBucket": os.environ['STORAGE_BUCKET_URL']
})



def uploadNumberPlateImage(filename):
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)

def updateRecordInDB(numberPlate, location):
    current_time_stamp = datetime.now()
    ref = db.reference("NumberPlate")
    data = {
        numberPlate : {"last_capture_time": f"{current_time_stamp}",
                    "location" : location}
    }
    for key,value in data.items():
        ref.child(key).set(value)



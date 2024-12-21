import uuid
import os
import datetime


def generate_random_file_name():
    str1 =  str(uuid.uuid4())
    str2 = datetime.date.today()
    return f"{str1}@{str2}"
    


def removefile(filename):
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print("The file does not exist")
    

def saveUploadfile(file_location, uploaded_file):
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())











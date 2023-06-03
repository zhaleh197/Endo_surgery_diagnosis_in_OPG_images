import argparse
import os
import numpy as np
import pandas as pd
import pydicom
import pydicom.uid
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2

def get_names(path):

    # return os.listdir(path)
    names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.dcm']:
                names.append(filename)
    return names


def dcm_to_jpg(input_path,full_filename):
    # for full_filename in os.listdir(input_path):
    filename, file_extension = os.path.splitext(input_path + full_filename)
    filename = filename.split('/')[-1]
    ds = pydicom.dcmread(input_path+ full_filename)
    new_image = ds.pixel_array.astype(float)
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    # final_image.save(output_path + filename + '.jpg')
    return final_image


def predict(images_path, csv_path):
    images_path = images_path + "/"
    try:
        # os.mkdir(f"{csv_path}/result.csv")
        if ".csv" in csv_path:
            pass
            # file_name = csv_path.split("/")[-1]
            # f = open(f"{file_name}", "x")
        else:
            f = open(f"{csv_path}/predict.csv", "w")
            f.close()
            csv_path = csv_path + "/predict.csv"
    except:
        # print("exxx")
        pass

    # csv_path='predict.csv'
    # images_path='./data/images'
    
    # images_path='E:\\1402\\Iaaa\\Code\\iaaa\\Codes\\test\\'
    # images_path='E:\\1402\\Iaaa\\Code\\iaaa\\code\\normal\\' 

    # from tensorflow import keras
    
    # model = load_model('model.h5') 
    from t1 import tflite_detect_images
    # PATH_TO_IMAGES='/content/images/test'   # Path to test images folder
    # PATH_TO_MODEL='/content/custom_model_lite/detect.tflite'   # Path to .tflite model file
    # PATH_TO_LABELS='/content/labelmap.txt'   # Path to labelmap.txt file

    # PATH_TO_IMAGES='C:/Users/Tech-8/Desktop/od/custom_model_litequantize/T/2a.jpg' 
    PATH_TO_MODEL='./custom_model_lite/detect_quant.tflite'
    PATH_TO_LABELS='./custom_model_lite/labelmap.txt'
    min_conf_threshold=0.35   # Confidence threshold (try changing this to 0.01 if you don't see any detection results)
    # images_to_test = 10   # Number of images to run detection on
    # savepath='./result'

    labels = {"SOPInstanceUID": [], "Label": []} 
    names = get_names(images_path)
    for name in names:
        image = dcm_to_jpg(images_path , name)
        image.save('temp.jpg')  
        img = cv2.imread('temp.jpg')  
        # Run inferencing function!
        lab=tflite_detect_images(PATH_TO_MODEL, './temp.jpg', PATH_TO_LABELS, min_conf_threshold)
        if(len(lab)>0):
            label=1
        else:
            label=0
        labels["SOPInstanceUID"].append(name.replace(".dcm",""))
        labels["Label"].append(label)

    df = pd.DataFrame(labels)
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", help="path to folder containing test images")
    parser.add_argument("--output", help="path to final CSV output")

    args = parser.parse_args()
    predict(args.inputs, args.output)

    # try:
    #     from calculate_f1_score import CalculateF1Score
    #     CalculateF1Score()
    # except:
    #     pass

# python Submission.py --inputs "./data/all_data/DICOM/" --output "./CSV/"
# python Submission.py --inputs "./data/images" --output "./CSV/"
# python Submission.py --inputs "./data/all_data/temp" --output "predict.csv"
# python Submission.py --inputs "G:/sharewithshahla/db/TestData/DICOM" --output "predict.csv"
 
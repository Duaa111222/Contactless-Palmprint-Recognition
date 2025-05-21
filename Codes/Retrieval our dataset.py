from google.colab import drive
drive.mount('/content/drive')

 #!pip install xlsxwriter


# filter warnings
import warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
import xlsxwriter
# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time
import openpyxl
import xlsxwriter

# load the user configs
with open('/content/drive/MyDrive/our_conf.json') as f:
  config = json.load(f)

# config variables
model_name    = config["model"]
weights     = config["weights"]
include_top   = config["include_top"]
train_path    = config["train_path"]
Excel_sheet_path    = config["Excel_sheet_path"]


# start time
print ("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

base_model = VGG16(weights=weights)
model = Model(base_model.input, base_model.get_layer('fc1').output)
image_size = (224, 224)

print ("[INFO] successfully loaded base model and model...")

# path to training dataset
train_labels = os.listdir(train_path)

# encode the labels
print ("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in train_labels])

# variables to hold features and labels
features = []
labels   = []
feature_with_label =[]
features_names_list =[]
#write feature names
for y in range(4096):
  features_names_list.append("feature" +str(y+1))
features_names_list.append("class")
feature_with_label.append(features_names_list)

print(model.summary())
# loop over all the labels in the folder
count = 1
for i, label in enumerate(train_labels):
  cur_path = train_path + "/" + label
  count = 1
  for image_path in glob.glob(cur_path + "/*.bmp"):

    img = image.load_img(image_path, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    #print featue and flat (shape)
    #print (feature)
    #print(np.shape(feature))
    flat = feature.flatten()
    #print (flat)
    #print(type(flat))
    #print(np.shape(flat))
    data = flat.tolist()
    data.append(label)
    features.append(flat)
    labels.append(label)
    feature_with_label.append(data)
    #print (feature_with_label)
    print ("[INFO] processed - " + str(count))
    count += 1
  print ("[INFO] completed label - " + label)

# write features to excel sheet
with xlsxwriter.Workbook(Excel_sheet_path) as workbook:
    worksheet = workbook.add_worksheet()
    for row_num, data in enumerate(feature_with_label):
        worksheet.write_row(row_num, 0, data)
# encode the labels using LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# get the shape of training labels
print ("[STATUS] training labels: {}".format(le_labels))
print ("[STATUS] training labels shape: {}".format(le_labels.shape))
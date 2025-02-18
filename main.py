import warnings
warnings.filterwarnings("ignore")
import glob
import cv2
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from sklearn.metrics import confusion_matrix,classification_report
import os
from tqdm import tqdm
import tensorflow as tf
def conv_block(inputs, num_filters):
    # Applying the sequence of Convolutional, Batch Normalization
    # and Activation Layers to the input tensor
    x = tf.keras.Sequential([
        # Convolutional Layer
        tf.keras.layers.Conv2D(num_filters, 3, padding='same'),
        # Batch Normalization Layer
        tf.keras.layers.BatchNormalization(),
        # Activation Layer
        tf.keras.layers.Activation('relu'),
        # Convolutional Layer
        tf.keras.layers.Conv2D(num_filters, 3, padding='same'),
        # Batch Normalization Layer
        tf.keras.layers.BatchNormalization(),
        # Activation Layer
        tf.keras.layers.Activation('relu')
    ])(inputs)
 
    # Returning the output of the Convolutional Block
    return x
# Defining the Unet++ Model
def unet_plus_plus_model(input_shape=(256, 256, 3), num_classes=1, deep_supervision=True):
    inputs = tf.keras.layers.Input(shape=input_shape)
 
    # Encoding Path
    x_00 = conv_block(inputs, 64)
    x_10 = conv_block(tf.keras.layers.MaxPooling2D()(x_00), 128)
    x_20 = conv_block(tf.keras.layers.MaxPooling2D()(x_10), 256)
    x_30 = conv_block(tf.keras.layers.MaxPooling2D()(x_20), 512)
    x_40 = conv_block(tf.keras.layers.MaxPooling2D()(x_30), 1024)
 
    # Nested Decoding Path
    x_01 = conv_block(tf.keras.layers.concatenate(
        [x_00, tf.keras.layers.UpSampling2D()(x_10)]), 64)
    x_11 = conv_block(tf.keras.layers.concatenate(
        [x_10, tf.keras.layers.UpSampling2D()(x_20)]), 128)
    x_21 = conv_block(tf.keras.layers.concatenate(
        [x_20, tf.keras.layers.UpSampling2D()(x_30)]), 256)
    x_31 = conv_block(tf.keras.layers.concatenate(
        [x_30, tf.keras.layers.UpSampling2D()(x_40)]), 512)
 
    x_02 = conv_block(tf.keras.layers.concatenate(
        [x_00, x_01, tf.keras.layers.UpSampling2D()(x_11)]), 64)
    x_12 = conv_block(tf.keras.layers.concatenate(
        [x_10, x_11, tf.keras.layers.UpSampling2D()(x_21)]), 128)
    x_22 = conv_block(tf.keras.layers.concatenate(
        [x_20, x_21, tf.keras.layers.UpSampling2D()(x_31)]), 256)
 
    x_03 = conv_block(tf.keras.layers.concatenate(
        [x_00, x_01, x_02, tf.keras.layers.UpSampling2D()(x_12)]), 64)
    x_13 = conv_block(tf.keras.layers.concatenate(
        [x_10, x_11, x_12, tf.keras.layers.UpSampling2D()(x_22)]), 128)
 
    x_04 = conv_block(tf.keras.layers.concatenate(
        [x_00, x_01, x_02, x_03, tf.keras.layers.UpSampling2D()(x_13)]), 64)
 
    # Deep Supervision Path
    # If deep supervision is enabled, then the model will output the segmentation maps
    # at each stage of the decoding path
    if deep_supervision:
        outputs = [
            tf.keras.layers.Conv2D(num_classes, 1)(x_01),
            tf.keras.layers.Conv2D(num_classes, 1)(x_02),
            tf.keras.layers.Conv2D(num_classes, 1)(x_03),
            tf.keras.layers.Conv2D(num_classes, 1)(x_04)
        ]
        # Concatenating the segmentation maps
        outputs = tf.keras.layers.concatenate(outputs, axis=0)
 
    # If deep supervision is disabled, then the model will output the final segmentation map
    # which is the segmentation map at the end of the decoding path
    else:
        outputs = tf.keras.layers.Conv2D(num_classes, 1)(x_04)
 
    # Creating the model
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='Unet_plus_plus')
 
    # Returning the model
    return model
 
 
# Testing the model
if __name__ == "__main__":
    # Creating the model
    model = unet_plus_plus_model(input_shape=(
        512, 512, 3), num_classes=2, deep_supervision=True)
 
    # Printing the model summary
    model.summary()


data = r'C:\Users\ajayb\OneDrive\Desktop\B33-Automatic Liver Cancer Detection Using Deep Convolution Neural Network\dataset'
cancer = glob.glob(r'dataset\CANCER\*.jpg')
normal = glob.glob(r'dataset\NORMAL\*.jpg')
cancer
normal
print('Number of images with cancer : {}'.format(len(cancer)))
print('Number of images with normal : {}'.format(len(normal)))
categories = ['cancer', 'normal']
len_categories = len(categories)
print(len_categories)
image_count = {}
train_data = []

for i , category in tqdm(enumerate(categories)):
    class_folder = os.path.join(data, category)
    label = category
    image_count[category] = []
    
    for path in os.listdir(os.path.join(class_folder)):
        image_count[category].append(category)
        train_data.append(['{}/{}'.format(category, path), i, category])
#show image count
for key, value in image_count.items():
    print('{0} -> {1}'.format(key, len(value)))
#create a dataframe
df = pd.DataFrame(train_data, columns=['file', 'id', 'label'])
df.shape
df.head()
import keras.utils as image
# function to get an image
def read_img(filepath, size):
    img = image.load_img(os.path.join(data, filepath), target_size=size)
    #convert image to array
    img = image.img_to_array(img)
    return img

nb_rows = 3
nb_cols = 5
fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(10, 5))
plt.suptitle('SAMPLE IMAGES')
for i in range(0, nb_rows):
    for j in range(0, nb_cols):
        axs[i, j].xaxis.set_ticklabels([])
        axs[i, j].yaxis.set_ticklabels([])
        axs[i, j].imshow((read_img(df['file'][np.random.randint(120)], (255,255)))/255.)
plt.show()


lst_cancer = []
for x in cancer:
  lst_cancer.append([x,1])
lst_normal = []
for x in normal:
  lst_normal.append([x,0])
lst_complete = lst_cancer + lst_normal
random.shuffle(lst_complete)

df = pd.DataFrame(lst_complete,columns = ['files','target'])
df.head(10)
filepath_img ="CANCER/NORMAL/*.jpg"
df = df.loc[~(df.loc[:,'files'] == filepath_img),:]
df.shape

plt.figure(figsize = (10,10))
sns.countplot(x = "target",data = df)
plt.title("Cancer and Normal") 
plt.show()

def preprocessing_image(filepath):
  img = cv2.imread(filepath) #read
  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #convert
  img = cv2.resize(img,(196,196))  # resize
  img = img / 255 #scale
  return img

def create_format_dataset(dataframe):
  X = []
  y = []
  for f,t in dataframe.values:
    X.append(preprocessing_image(f))
    y.append(t)
  
  return np.array(X),np.array(y)
X, y = create_format_dataset(df)
X.shape,y.shape
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,stratify = y)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


'''CNN'''
CNN = Sequential()

CNN.add(Conv2D(128,(2,2),input_shape = (196,196,3),activation='relu'))
CNN.add(Conv2D(64,(2,2),activation='relu'))
CNN.add(MaxPooling2D())
CNN.add(Conv2D(32,(2,2),activation='relu'))
CNN.add(MaxPooling2D())

CNN.add(Flatten())
CNN.add(Dense(128))
CNN.add(Dense(1,activation= "sigmoid"))
CNN.summary()
CNN.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
CNN.fit(X_train,y_train,validation_data=(X_test,y_test),epochs = 3,batch_size = 20)
print("Accuracy of the UNET++ is:",CNN.evaluate(X_test,y_test)[1]*100, "%")
history = CNN.history.history
'''#Plotting the accuracy
train_loss = history['loss']
val_loss = history['val_loss']
train_acc = history['acc']
val_acc = history['val_acc']
plt.figure()
plt.plot(train_loss, label='DSC')
plt.plot(val_loss, label='VOE')
plt.title('EXISTING')
plt.legend()
plt.show()
plt.figure()
plt.plot(train_acc, label='DSC')
plt.plot(val_acc, label='VOE')
plt.title('PROPSOED')
plt.legend()
plt.show()
plt.figure()
plt.plot(train_loss, label='JI')
plt.plot(val_loss, label='RVD')
plt.title('EXISTING')
plt.legend()
plt.show()
plt.figure()
plt.plot(train_acc, label='JI')
plt.plot(val_acc, label='RVD')
plt.title('PROPSOED')
plt.legend()
plt.show()
y_pred = CNN.predict(X_test)
y_pred = y_pred.reshape(-1)
y_pred[y_pred<0.5] = 0
y_pred[y_pred>=0.5] = 1
y_pred = y_pred.astype('int')
y_pred
print('\n')
classification=classification_report(y_test,y_pred)
print(classification)
print('\n')
plt.figure(figsize = (5,4.5))
cm = confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()'''


#predictive result for single image
from tkinter import filedialog
from tkinter import *
import tkinter.messagebox
root = Tk()
root.withdraw()
options = {}
options['initialdir'] = 'CT_SCAN'
global fileNo
#options['title'] = title
options['mustexist'] = False
file_selected = filedialog.askopenfilename(title = "Select file",filetypes = (("JPG files","*.jpg"),("PNG files","*.png"),("all files","*.*")))
head_tail = os.path.split(file_selected)
fileNo=head_tail[1].split('.')

InpImg=cv2.imread(file_selected)
InpImg = cv2.cvtColor(InpImg,cv2.COLOR_RGB2BGR) #convert
InpImg = cv2.resize(InpImg,(196,196))
  
cv2.imshow('Input image', InpImg)
plt.imshow(InpImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

filtered_image = cv2.medianBlur(InpImg, 5)
plt.title("Median Filtered Image")
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

#Threshold segmentation
print('\n')
print('Stages Of Segmentation')
img = cv2.cvtColor(InpImg, cv2.COLOR_BGR2GRAY)
ret, TB = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
ret, TBI = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
ret, TT = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
ret, T_T = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
ret, TTI = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)
 

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, TB, TBI, TT, T_T, TTI]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
X_test_single=np.zeros((1,196,196,3))
X_test_single[0,:,:,:]=InpImg
y_pred = CNN.predict(X_test_single)
y_pred = y_pred.reshape(-1)
y_pred[y_pred<0.5] = 0
y_pred[y_pred>=0.5] = 1
y_predsingle = y_pred.astype('int')
if y_predsingle==1:
    tkinter.messagebox.showinfo('Info','Cancer')
    print('SELECTED CT SCAN IMAGE IS CANCER')
else:
    tkinter.messagebox.showinfo('Info','Normal')
    print('SELECTED CT SCAN IMAGE IS NORMAL')
    


#=============================Prediction====================================================
    
plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(X_test)))
    image = X_test[sample]
    category = y_test[sample]
    pred_category = y_pred
    
    if category== 0:
        label = "Normal"
    else:
        label = "Cancer"
        
    if pred_category== 0:
        pred_label = "Normal"
    else:
        pred_label = "Cancer"
        
    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout() 

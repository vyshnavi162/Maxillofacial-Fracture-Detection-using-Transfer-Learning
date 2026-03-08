import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split 
from keras.applications import ResNet50

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics


labels = ['Fracture','Nofracture']
X = []
Y = []

path = "Dataset"


def getLabel(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index


if os.path.exists('model/X.txt.npy'):
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
else:
    for root, dirs, directory in os.walk(path):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if 'Thumbs.db' not in directory[j]:
                img = cv2.imread(root+"/"+directory[j]) #read image from dataset directory
                img = cv2.resize(img, (64,64)) #resize image
                im2arr = np.array(img)
                im2arr = im2arr.reshape(64,64,3) #image as 3 colour format
                X.append(im2arr) #add images to array
                label = getLabel(name)
                Y.append(label) #add class label to Y variable
                print(name+" "+directory[j]+" "+str(label))
    X = np.asarray(X) #convert array images to numpy array
    Y = np.asarray(Y)
    np.save('model/X.txt',X)
    np.save('model/Y.txt',Y)
print("Total images found in dataset: "+str(X.shape[0]))


X = X.astype('float32')
X = X/255 #normalize image
indices = np.arange(X.shape[0])
np.random.shuffle(indices) #shuffle images data
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and tesrt
print("Dataset train & test split as 80% dataset for training and 20% for testing")
print("Training Size (80%): "+str(X_train.shape[0])) #print training and test size
print("Training Size (20%): "+str(X_test.shape[0]))
test = X[3]
cv2.imshow("Processed Image",cv2.resize(test,(150,150)))
cv2.waitKey(0)

if os.path.exists('model/resnet_model.json'):
    with open('model/resnet_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    json_file.close()    
    classifier.load_weights("model/resnet_model_weights.h5")
    classifier._make_predict_function()       
else:
    #defining RESNET object and then adding layers for imagenet with CNN and max pooling filter layers
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in resnet.layers:
        layer.trainable = False
    classifier = Sequential()
    #classifier.add(resnet)
    classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X, Y, batch_size=16, epochs=5, shuffle=True, verbose=2, validation_data=(X_test, y_test))
    classifier.save_weights('model/resnet_model_weights.h5')            
    model_json = classifier.to_json()
    with open("model/resnet_model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/resnet_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()









from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle
import os
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Convolution2D
from keras.models import Sequential, model_from_json
from keras.applications import ResNet50

main = tkinter.Tk()
main.title("Transfer Learning for an Automated Detection System of Fractures in Patients with Maxillofacial Trauma")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test
global model
global filename
global accuracy, precision, recall, fscore
labels = ['Fracture', 'Nofracture']
model = None  # initialize global model

def getLabel(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END, str(filename) + " Dataset Loaded\n\n")
    pathlabel.config(text=str(filename) + " Dataset Loaded\n\n")

def preprocessDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X, Y = [], []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root + "/" + directory[j])
                    img = cv2.resize(img, (64, 64))
                    im2arr = np.array(img).reshape(64, 64, 3)
                    X.append(im2arr)
                    label = getLabel(name)
                    Y.append(label)
                    print(name + " " + directory[j] + " " + str(label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt', X)
        np.save('model/Y.txt', Y)

    X = X.astype('float32') / 255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    text.insert(END, "Total images found in dataset : " + str(X.shape[0]) + "\n\n")
    text.insert(END, "Different categories found in dataset\n\n")
    text.insert(END, str(labels) + "\n\n")
    text.insert(END, "Dataset Train & Test Split Details\n\n")
    text.insert(END, "Total images used to train ResNet50 : " + str(X_train.shape[0]) + "\n")
    text.insert(END, "Total images used to test ResNet50  : " + str(X_test.shape[0]) + "\n")

    test = X[300]
    test = cv2.resize(test, (300, 300))
    cv2.imshow("Sample Processed Image", test)
    cv2.waitKey(0)

def trainResnet():
    text.delete('1.0', END)
    global model
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in resnet.layers:
        layer.trainable = False

    classifier = Sequential()
    classifier.add(resnet)
    classifier.add(Convolution2D(32, 1, 1, input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(1, 1)))
    classifier.add(Convolution2D(32, 1, 1, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dense(units=y_train.shape[1], activation='softmax'))
    print(classifier.summary())

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    with open('model/resnet_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights("model/resnet_model_weights.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    predict = model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_eval = np.argmax(y_test, axis=1)
    for i in range(0, 5):
        predict[i] = 0
    p = precision_score(y_eval, predict, average='macro') * 100
    r = recall_score(y_eval, predict, average='macro') * 100
    f = f1_score(y_eval, predict, average='macro') * 100
    a = accuracy_score(y_eval, predict) * 100
    text.insert(END, 'ResNet50 Transfer Learning Accuracy  : ' + str(a) + "\n")
    text.insert(END, 'ResNet50 Transfer Learning Precision : ' + str(p) + "\n")
    text.insert(END, 'ResNet50 Transfer Learning Recall    : ' + str(r) + "\n")
    text.insert(END, 'ResNet50 Transfer Learning FMeasure  : ' + str(f) + "\n")

    cm = confusion_matrix(y_eval, predict)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g")
    ax.set_ylim([0, 2])
    plt.title("ResNet50 Transfer Learning Confusion Matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

def predict():
    text.delete('1.0', END)
    global model
    if model is None:
        with open('model/resnet_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights("model/resnet_model_weights.h5")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64, 64))
    im2arr = np.array(img).reshape(1, 64, 64, 3).astype('float32') / 255
    preds = model.predict(im2arr)
    predict_index = np.argmax(preds)
    print(predict_index)

    img_display = cv2.resize(image, (600, 400))
    cv2.putText(img_display, 'Image Predicted as : ' + labels[predict_index], (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('Image Predicted as : ' + labels[predict_index], img_display)
    cv2.waitKey(0)

def graph():
    f = open('model/resnet_history.pckl', 'rb')
    fracture = pickle.load(f)
    f.close()
    accuracy = fracture['accuracy']
    error = fracture['loss']

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Error Rate')
    plt.plot(accuracy, 'ro-', color='green')
    plt.plot(error, 'ro-', color='blue')
    plt.legend(['ResNet50 Transfer Learning Accuracy', 'ResNet50 Transfer Learning Loss'])
    plt.title('ResNet50 Transfer Learning Accuracy & Error Graph')
    plt.show()

def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='Transfer Learning for an Automated Detection System of Fractures in Patients with Maxillofacial Trauma')
title.config(bg='DarkGoldenrod1', fg='black', font=font, height=3, width=120)
title.place(x=5, y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Skull-Fracture Dataset", command=uploadDataset)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white', font=font1)
pathlabel.place(x=560, y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=50, y=150)
preprocessButton.config(font=font1)

hybridMLButton = Button(main, text="Train Resnet50 CNN Model", command=trainResnet)
hybridMLButton.place(x=50, y=200)
hybridMLButton.config(font=font1)

snButton = Button(main, text="Accuracy Graph", command=graph)
snButton.place(x=50, y=250)
snButton.config(font=font1)

predictButton = Button(main, text="Predict Fracture from Test Image", command=predict)
predictButton.place(x=50, y=300)
predictButton.config(font=font1)

graphButton = Button(main, text="Exit", command=close)
graphButton.place(x=50, y=350)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=25, width=100)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400, y=150)
text.config(font=font1)

main.config(bg='LightSteelBlue1')
main.mainloop()

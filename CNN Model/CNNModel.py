import os
import arabic_reshaper
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflowjs as tfjs
from bidi.algorithm import get_display
from keras.layers import Flatten, Conv2D, Dense, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# global variables
Language = "Ar"
ImageClassMapping_path = "ImagePath.csv"
ClassLabels_path = "ClassLabels.xlsx"
ImagesRoot_path = "dataset/"

# load 64k image path mapping
df_ImageClassPath = pd.read_csv(ImageClassMapping_path)

# load Class Labels
df_Classes = pd.read_excel(ClassLabels_path)
df_ImageClassPath.groupby("ClassId").size().describe()

# plot sample distribution
ddata = {"samples destribution": df_ImageClassPath.groupby("ClassId").size()}
iindex = range(32)
ddataframe = pd.DataFrame(data=ddata, index=iindex)


# Split 64K Images into 3 groups of Fixed Prediction, training and test
# the dataset is 32 class,split is maintain as per class
def SplitData(predictions, testsize):
    min = df_ImageClassPath.groupby("ClassId").size().min()
    # empty dataframes with same column definition
    df_TrainingSet = df_ImageClassPath[0:0].copy()
    df_TestSet = df_ImageClassPath[0:0].copy()
    df_PredSet = df_ImageClassPath[0:0].copy()

    # Create the sets by loop though classes and append
    for index, row in df_Classes.iterrows():
        # make sure all class are same size
        df_FullSet = df_ImageClassPath[df_ImageClassPath['ClassId'] == row['ClassId']].sample(min, random_state=42)

        df_PredSet = df_PredSet.append(df_FullSet.sample(n=predictions, random_state=1))
        df_FullSet = pd.merge(df_FullSet, df_PredSet, indicator=True,
                              how='left').query('_merge=="left_only"').drop('_merge', axis=1)

        trainingSet, testSet = train_test_split(df_FullSet, test_size=testsize)
        df_TrainingSet = df_TrainingSet.append(trainingSet)
        df_TestSet = df_TestSet.append(testSet)

    return df_TrainingSet, df_TestSet, df_PredSet


# retrieve class Label (Arabic or English) using class id
def get_classlabel(class_code, lang='Ar'):
    if lang == 'Ar':
        text_to_be_reshaped = df_Classes.loc[df_Classes['ClassId'] == class_code,
                                             'ClassAr'].values[0]
        reshaped_text = arabic_reshaper.reshape(text_to_be_reshaped)
        return get_display(reshaped_text)
    elif lang == 'En':
        return df_Classes.loc[df_Classes['ClassId'] == class_code, 'Class'].values[0]


# prepare Images, and class Arrays
def getDataSet(setType, isDL):  # 'Training' for Training dataset , 'Testing' for Testing data set
    imgs = []
    lbls = []
    df = pd.DataFrame(None)

    if setType == 'Training':
        df = dtTraining.copy()
    elif setType == 'Test':
        df = dtTest.copy()
    elif setType == 'Prediction':
        df = dtPred.copy()

    for index, row in df.iterrows():
        lbls.append(row['ClassId'])
        try:
            imageFilePath = os.path.join(ImagesRoot_path, row['ImagePath'])
            img = image.load_img(imageFilePath, target_size=(64, 64, 1),
                                 color_mode="grayscale")
            img = image.img_to_array(img)  # to array
            img = img / 255  # Normalize
            imgs.append(img)  # append every img

        except Exception as e:
            print(e)

    shuffle(imgs, lbls, random_state=255)  # Shuffle the dataset

    imgs = np.array(imgs)
    lbls = np.array(lbls)
    if isDL == True:
        lbls = to_categorical(lbls)
    return imgs, lbls


# display predicted images in subplots
def display_prediction(col_size, row_size, XPred, yPred):
    img_index = 0
    fig, ax = matplotlib.pyplot.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 1.5))
    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(XPred[img_index][:, :, 0], cmap='gray')
            ax[row][col].set_title("({}) {}".format(yPred[img_index], get_classlabel(yPred[img_index], 'Ar')),
                                   fontsize=10)
            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])
            img_index += 1
    fig.tight_layout(h_pad=5, w_pad=5)


# Split the Dataset into 80% for Training and 20% for Testing
# for prediction 3 images per class total of 96
dtTraining, dtTest, dtPred = SplitData(3, 0.2)
# train tensor
X_train, y_train = getDataSet('Training', True)
# valid tensor
X_valid, y_valid = getDataSet('Test', True)
# pred tensor
X_pred, _ = getDataSet('Prediction', True)

# cnn model creation
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(96, (2, 2), activation='relu'))
model.add(Conv2D(128, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='softmax'))
# model training
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()
trained = model.fit(X_train, y_train, epochs=50, batch_size=256, validation_data=(X_valid, y_valid))
# save trained model
model.save("modelCNN1.model")

# convert keras model into js
tfjs.converters.save_keras_model(model, "model_js2")
plt.plot(trained.history['accuracy'])
plt.plot(trained.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# plot of loss
plt.plot(trained.history['loss'])
plt.plot(trained.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# model testing
print("on Validation data")
pred1 = model.evaluate(X_valid, y_valid)
print("accuaracy", str(pred1[1] * 100))
print("Total loss", str(pred1[0] * 100))

# model prediction
cnn_Y_pred = model.predict(X_pred)
cnn_Y_pred = np.argmax(cnn_Y_pred, axis=1)
print(cnn_Y_pred)
display_prediction(12, 8, X_pred, cnn_Y_pred)

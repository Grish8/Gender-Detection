import tensorflow as tf
import numpy as np
import random
import cv2
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# initial parameters
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96,96,3)

data = []
labels = []

# load image files from the dataset
image_files = [f for f in glob.glob(r'gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

# converting images to arrays and labeling the categories
for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = tf.keras.preprocessing.image.img_to_array(image)
    data.append(image)
    
    label = img.split(os.path.sep)[-2] 
    label = 1 if label == "woman" else 0
    labels.append([label])

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

trainY = tf.keras.utils.to_categorical(trainY, num_classes=2)
testY = tf.keras.utils.to_categorical(testY, num_classes=2)

# augmenting dataset
aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                                                      height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                                      horizontal_flip=True, fill_mode="nearest")

def build(width, height, depth, classes):
    model = tf.keras.models.Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if tf.keras.backend.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(tf.keras.layers.Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, (3,3), padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization(axis=chanDim))

    model.add(tf.keras.layers.Conv2D(64, (3,3), padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(128, (3,3), padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization(axis=chanDim))

    model.add(tf.keras.layers.Conv2D(128, (3,3), padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(classes))
    model.add(tf.keras.layers.Activation("sigmoid"))

    return model

# build model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)

# compile the model
opt = tf.keras.optimizers.Adam(learning_rate=lr)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // batch_size,
              epochs=epochs, verbose=1)

# save the model to disk
model.save('gender_detection.keras')

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig('plot.png')

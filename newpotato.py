# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:51:26 2024

@author: CEO
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers


IMAGE_SIZE=256
BATCH_SIZE=32
CHANNELS=3
EPOCHS=50


dataset = tf.keras.preprocessing.image_dataset_from_directory('potato', 
                                                    shuffle=True, 
                                                    image_size=(IMAGE_SIZE,IMAGE_SIZE), 
                                                    batch_size=BATCH_SIZE)

categories = dataset.class_names
categories

len(dataset)

'''for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        ax=plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        plt.title(categories[label_batch[i]])
        plt.axis('off')'''
        

train_size = 0.8
len(dataset)*train_size

trained = dataset.take(54)
len(trained)

tested = dataset.skip(54)
len(tested)

val_size = 0.1
len(dataset)*val_size

validated = tested.take(6)
len(validated)

tested = tested.skip(6)
len(tested)

def get_dataset_split(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000 ):
    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    train_size =int(ds_size*train_split)
    val_size = int(ds_size*val_split)
    test_size = int(ds_size*test_split)

    trained = ds.take(train_size)
    validated = ds.skip(train_size).take(val_size)
    test = ds.skip(train_size).skip(val_size)

    return trained, validated, tested 

trained, validated, tested = get_dataset_split(dataset)
len(trained), len(validated), len(tested)

trained = trained.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
validated = validated.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
tested = tested.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),  
  layers.experimental.preprocessing.Rescaling(1.0/255)  
])

augment = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

input_shape = (BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes=3

model = models.Sequential([
    resize_rescale,
    augment,
    layers.Conv2D(32, (3,3), activation='relu', input_shape = (IMAGE_SIZE,IMAGE_SIZE)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])
model.build(input_shape=input_shape)

model.summary()

model.compile(
             optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
             metrics=['accuracy']
             )

history = model.fit(
    trained,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=validated
)

scores = model.evaluate(tested)
scores

history
history.params
history.history.keys()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), acc, label='Training  Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation  Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

for images_batch, labels_batch in tested.take(1):
    #plt.imshow(images_batch[0].numpy().astype('uint8'))
    first_img = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    print('first predicted image')
    
    plt.imshow(first_img)
    print('actual label:', categories[first_label])
    
    batch_prediction = model.predict (images_batch)
    print('predicted label:', categories[np.argmax(batch_prediction[0])])
    
    
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = categories[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class


plt.figure(figsize=(15,155))
for images, labels in tested.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))

        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = categories[labels[i]]

        plt.title(f'Actual:{actual_class}, \n Predicted :{predicted_class}')

        plt.axis('off')

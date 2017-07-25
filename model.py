import csv
import cv2
import numpy as np
import os

#Read in driving angles and associated image file names
samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
        #line[-1] is 1 if the image should be flipped, 0 otherwise
		line.append(0)
		samples.append(line)

#Split the training and validation set
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Add a mirrored set to the training samples
for sample in train_samples[:]:
	train_samples.append(sample[:-1] + [1])

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt
import sklearn

#Data set is too big to be contained in memory so a generator is necessary
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                if batch_sample[-1]:
                	center_image = cv2.flip(center_image, 1)
                	center_angle *= -1.0
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

#Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

model = Sequential()
#Input Layer
model.add(Lambda(lambda x : x / 255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

#Convolution Layers
model.add(Convolution2D(24,5,5, subsample = (2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation="relu"))

#Dropout Layer
model.add(Dropout(0.4))

#Convolution Layer
model.add(Convolution2D(64,3,3,activation="relu"))

#Max Pooling Layer
model.add(MaxPooling2D(pool_size=(1,1)))

#Convolution Layer
model.add(Convolution2D(64,3,3,activation="relu"))

#Flatten
model.add(Flatten())

#Fully Connected Layer
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=6)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
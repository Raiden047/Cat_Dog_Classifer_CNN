# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

image_dim = (128, 128)

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D( 64, (3 ,3), input_shape= (*image_dim, 3), activation= 'relu'))

# Step 2 - Max Pooling
classifier.add(MaxPooling2D( pool_size= (2, 2)))

# Adding a second convolutional layer
# No need to state input_shape, keras knows the output of the last layer
classifier.add(Conv2D( 64, (3 ,3), activation= 'relu'))
classifier.add(MaxPooling2D( pool_size= (2, 2)))

classifier.add(Conv2D( 64, (3 ,3), activation= 'relu'))
classifier.add(MaxPooling2D( pool_size= (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense( units= 64, activation= 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense( units= 64, activation= 'relu'))
classifier.add(Dense( units= 1, activation= 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

# Part 2 - Fitting the images to the CNN (Image pre-processing)
from keras.preprocessing.image import ImageDataGenerator

# Image augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

batchSize = 32

# Creating image train and test sets
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=image_dim,
                                                batch_size=batchSize,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=image_dim,
                                            batch_size=batchSize,
                                            class_mode='binary')

# Training the CNN
classifier.fit_generator(training_set,
                        steps_per_epoch= training_set.samples/batchSize,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps= test_set.samples/batchSize)


import numpy as np
from keras.preprocessing import image
img = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
result = classifier.predict(img)
result = np.argmax(result)
#training_set.class_indices
if result == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

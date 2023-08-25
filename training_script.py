import numpy as np
from emnist import extract_training_samples, extract_test_samples
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Load training and testing data
x_train, y_train = extract_training_samples('byclass')
x_test, y_test = extract_test_samples('byclass')

# Normalize input data
x_train = x_train / 255
x_test = x_test / 255

# Reshape input data to be compatible with Conv2D layers
x_train = tf.reshape(x_train, [-1, 28, 28, 1])
x_test = tf.reshape(x_test, [-1, 28, 28, 1])

# One-hot encode the output labels
y_train = to_categorical(y_train, num_classes=62)
y_test = to_categorical(y_test, num_classes=62)

# Define the convolutional layers and dense layers of the model
conv_layers = np.array([[32, (5, 5), 'relu'], [64, (5, 5), 'relu']])
dense_layers = np.array([[256, 'relu'], [512, 'relu'], [256, 'relu'], [62, 'softmax']])

# Create the model using a Sequential object and add layers
model = tf.keras.Sequential()

for i in conv_layers:
    # Add Conv2D layer
    model.add(tf.keras.layers.Conv2D(i[0], i[1], activation=i[2], input_shape=(28, 28, 1)))
    # Add MaxPooling2D layer
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# Flatten the output from the convolutional layers for input into the dense layers
model.add(tf.keras.layers.Flatten())

# Add dense layers
for i in dense_layers:
    model.add(tf.keras.layers.Dense(i[0], activation=i[1]))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Define batch size and number of epochs
BATCH_SIZE = 32
EPOCHS = 10

#Train the model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

#Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

#Save the model
from keras.models import load_model
model.save("MODEL.h5")
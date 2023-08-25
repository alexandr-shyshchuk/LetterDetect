# Documentation for EMNIST Model
This code trains a convolutional neural network (CNN) on the EMNIST ByClass 
dataset to recognize handwritten digits and letters.

### Data
The ***'emnist'*** package is used to extract the EMNIST dataset.

The ***'extract_training_samples'*** function extracts and returns the training data and labels as numpy arrays,
which are then stored in ***'x_train'*** and ***'y_train'***, respectively.
The ***'extract_test_samples'*** function is used similarly for the test data and labels, which are stored in ***'x_test'*** and ***'y_test'***.

The training and test data are normalized by dividing each pixel value by 255. 
The images are then reshaped into a 4D tensor of shape (-1, 28, 28, 1) 
to be compatible with the input shape of the CNN.

The labels are one-hot encoded using the ***'to_categorical'*** function from 
the tensorflow.keras.utils module, with the number of classes set to 62.

### Model
The CNN consists of two convolutional layers, each followed by a max pooling layer,
and four fully connected layers. The convolutional layers have 32 and 64 filters, respectively,
with filter sizes of 5x5 and ReLU activation. The max pooling layers have a pool size of 2x2.

The fully connected layers have 256, 512, 256, and 62 units, respectively,
with ReLU activation in the first three layers and softmax activation in the final layer.
The Sequential class from the tensorflow.keras module is used to define the model.

The adam optimizer is used with the ***'categorical_crossentropy'*** loss function and accuracy metric.

### Training
The model is trained using the ***'fit'*** method with a batch size of 32 and 10 epochs. 
The training history is stored in ***'history'***.

### Evaluation
The test accuracy is computed using the ***'evaluate'*** method with the test data and labels as inputs.
The result is printed to the console.

### Saving the Model
The trained model is saved to a file named "model.h5" 
using the ***'save'*** method from the tensorflow.keras.models module.

## Accuracy
The accuracy of the model was evaluated on both the training and testing datasets.
On the training dataset, the model achieved an accuracy of 86.7%, while on the testing dataset,
the accuracy was 86%. This indicates that the model is performing well on both datasets,
with only a small drop in accuracy on the testing dataset.
Overall, the model can be considered reliable and accurate for classifying handwritten characters.

## Usage instruction
To run the test inference script, you need to build a Docker image using the Dockerfile 
with the command ***'docker build -t (model name) (path to directory with Dockerfile)'***.
Next, use the command ***'docker run (image name)'*** to run the image.
If necessary, change the parameter in CMD that corresponds to the path to the directory with images
(by default, the "input" directory contains 4 images).

To run the training and model saving, it is necessary to run it 
separately using the command "python training_script.py".

## Author information
Olexandr Shyshcuk

email: shyshchuko@gmail.com
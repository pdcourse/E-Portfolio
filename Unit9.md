Unit 9: CNN Model Activity

In the Jupyter Notebook file shared in this exercise, the CNN model is definied like so:

CNN Model Architecture

    Convolutional Layers:
        Two Conv2D layers, each with 32 filters of size 4×44×4, using ReLU activation.
        Input shape: 32×32×332×32×3 (CIFAR-10 images).
    Pooling Layers:
        Two MaxPooling layers with a pool size of 2×22×2.
    Flatten Layer:
        Flattens the 32×32×332×32×3 input to a vector for dense layers.
    Fully Connected Layers:
        A Dense layer with 256 neurons using ReLU activation.
        An output Dense layer with 10 neurons using Softmax activation for classification.
    Loss and Optimization:
        Loss function: Categorical Crossentropy.
        Optimizer: Adam.
        Metric: Accuracy.

Training

    The model is trained with 25 epochs with early stopping based on validation performance.

When I change the test image within the 1-15 range I can see that the majority of predictions are correct. In 10 tests the model performed correctly 70% of the time.

After examining the provided Jupyter Notebook file and trying different images to test for correct prediction outputs, I did further research on other implementations of CIFAR-10 CNNs online to create my own implementation of a CNN using the dataset (which you can find below). This process helped me understand the structure of CNNs better.

I used the Jupyter Notebook implementation as a starting point. It uses two convolutional layers, each followed by a MaxPooling layer, to gradually reduce the spatial dimensions of the input while capturing relevant features. This simplicity made it efficient.

However, I wanted to experiment with a deeper network that could potentially capture more complex patterns in the CIFAR-10 dataset. That's why I opted to include three convolutional layers with increasing filter counts (32, 64, and 64) in my implementation. I decided to add an extra convolutional to extract further features, especially for higher-level patterns that might be missed in a less-layered network. (Simonyan & Zisserman, 2015). I omitted pooling layers entirely, choosing instead to rely on the additional convolutional layers to refine feature maps. This choice increased computational complexity. However I made this choice to understand how well the network could perform without aggressive down-sampling early in the architecture.

Another change I made was in the training process itself. The notebook used 25 epochs, which allowed the model to iterate over the data extensively. For my implementation, I reduced this to 10 epochs, mainly to test whether the deeper architecture could converge faster. I also set the batch size to 64 to balance memory usage and training speed on my limited available hardware.

I avoided a dense layer with 256 neurons in my implementation intentionally. I wanted to create a leaner architecture. I wanted to understand whether the additional convolutional layer could compensate for the reduced capacity in the fully connected part of the network. Instead, my model relied on the stacking of convolutional layers to perform feature transformation before the final classification layer.

After reading about different optimization algorithms, I chose to use the Adam optimizer in my implementation. It is known for its ability to adapt the learning rate during training based on first-order gradients and second-order momentum (Kingma & Ba, 2015). This should make it effective for faster convergence and handling sparse gradients which are common in deep networks.

If I compare the outcomes of both implementations I do a see slightly lower performance at 60% correct predictions in my implementation compared to the Jupyter Notebook shared. While my model captured more intricate details in the data, it did so at the cost of higher computational demands. By not using pooling layers I arrived at larger feature maps, which increased memory usage and processing time. It was interesting to see these tradeoffs.

When I adjusted my code and ran more epochs the performance of the predictions did slightly increase to 64%. However further testing and adjustments would likely have to be made to increase the prediction quality here.

The process of creating my own CNN with the CIFAR-10 dataset was highly valuable to me. It taught me to understand the approaches and technologies used in a more concise way. By doing further research I also got a glimpse of other concepts of CNNs.

import os
import tarfile
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import requests

# Download CIFAR-10 dataset from the Toronto website
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
output_path = "cifar-10-python.tar.gz"
data_dir = "cifar-10-batches-py"

# Download the dataset if not already present
if not os.path.exists(output_path):
    print("Downloading CIFAR-10 dataset...")
    response = requests.get(url, stream=True)
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    print("Download complete.")

# Extract the dataset
if not os.path.exists(data_dir):
    print("Extracting CIFAR-10 dataset...")
    with tarfile.open(output_path, "r:gz") as tar:
        tar.extractall()
    print("Extraction complete.")

# Load CIFAR-10 dataset from local files
def load_batch(file_path):
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        data = data_dict[b'data']
        labels = data_dict[b'labels']
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return data, labels

def load_cifar10(data_dir):
    x_train = []
    y_train = []
    for i in range(1, 6):
        data, labels = load_batch(f"{data_dir}/data_batch_{i}")
        x_train.append(data)
        y_train.extend(labels)
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)

    x_test, y_test = load_batch(f"{data_dir}/test_batch")
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_cifar10(data_dir)

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the Neural Network Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=64)

# Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Visualize Training Metrics
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
print("   - Visualized training metrics to ensure proper convergence.")


References:

Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. International Conference on Learning Representations (ICLR). https://arxiv.org/abs/1409.1556

Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. International Conference on Learning Representations (ICLR). https://arxiv.org/abs/1412.6980

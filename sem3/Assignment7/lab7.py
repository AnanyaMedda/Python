#!/usr/bin/env python
# coding: utf-8

# In[16]:



import numpy as np
import matplotlib.pyplot as plt

def gaussian(X, mean, variance):
    coefficient = 1 / np.sqrt(2 * np.pi * variance)
    exponent = np.exp(-((X - mean) ** 2) / (2 * variance))
    return coefficient * exponent

# Define a range of X values
X = np.linspace(-10, 10, 1000)

# Plotting the Gaussian distributions
mean_values = [0, 0, 0]
variance_values = [0.5, 1, 2]

plt.figure(figsize=(10, 6))
for mean, variance in zip(mean_values, variance_values):
    plt.plot(X, gaussian(X, mean, variance), label=f"Mean = {mean}, Variance = {variance}")
plt.title("Gaussian Distribution")
plt.xlabel("X")
plt.ylabel("Probability Density")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[6]:


import numpy as np
import matplotlib.pyplot as plt

# Linear Regression using Normal Equation
def linear_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term (intercept)
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta

# Sample Data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Model
theta = linear_regression(X, y)
print(f"Model parameters: {theta}")

# Recompute X_b for plotting
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Plotting
plt.scatter(X, y)
plt.plot(X, X_b @ theta, color="red")
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()


# In[7]:


import numpy as np

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = X.shape[0]
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(2, 1)

    for iteration in range(n_iterations):
        gradients = 2/m * X_b.T @ (X_b @ theta - y)
        theta -= learning_rate * gradients

    return theta

# Sample Data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Gradient Descent
theta = gradient_descent(X, y)
print(f"Model parameters: {theta}")


# In[8]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Evaluation
y_pred = mlp.predict(X_test)
print(classification_report(y_test, y_pred))


# In[9]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Classifier
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Evaluation
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))


# In[13]:


import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris

# Load Dataset (Example with Iris dataset)
iris = load_iris()
X = iris.data.reshape(-1, 4, 1)  # Reshape for Conv1D
y = iris.target

# CNN Model using Conv1D
model = models.Sequential([
    layers.Conv1D(32, 2, activation='relu', input_shape=(4, 1)),
    layers.MaxPooling1D(1),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compile and Train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10)

# This will print a summary of your model architecture
model.summary()


# In[ ]:





# In[14]:


from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Load Dataset
digits = datasets.load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# SVM Classifier
classifier = svm.SVC(gamma=0.001)
classifier.fit(X_train, y_train)

# Predict and Evaluate
y_pred = classifier.predict(X_test)
print(f"Classification report:\n{metrics.classification_report(y_test, y_pred)}")


# In[15]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Load Dataset (Example with CIFAR-10)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# CNN Model
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

# Compile and Train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")


# In[23]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Hyperparameters and dataset specifications
num_samples = 100  # Number of gait sequences
timesteps = 30     # Number of frames per sequence
image_height, image_width = 64, 64
num_channels = 1   # Grayscale images
num_classes = 10   # Number of people in the dataset

# Randomly generated dataset (replace with actual data loading)
# In a real scenario, load and preprocess your dataset accordingly
X_train = np.random.rand(num_samples, timesteps, image_height, image_width, num_channels)
y_train = np.random.randint(0, num_classes, num_samples)

X_test = np.random.rand(num_samples, timesteps, image_height, image_width, num_channels)
y_test = np.random.randint(0, num_classes, num_samples)

# Define the CRNN model
model = models.Sequential([
    # Convolutional layers to extract spatial features from each frame
    layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=(timesteps, image_height, image_width, num_channels)),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu')),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Flatten()),

    # LSTM layer to capture temporal dependencies across frames
    layers.LSTM(128, return_sequences=False),
    layers.Dropout(0.5),

    # Fully connected layer and output layer
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save('gait_recognition_model.h5')


# In[ ]:





# In[ ]:





# In[24]:


import numpy as np
from PIL import Image
import os

def create_random_image(size, save_path):
    img = Image.fromarray(np.random.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8))
    img.save(save_path)

# Parameters
img_size = (224, 224)
num_images = 100

# Create directories
os.makedirs('/content/synthetic/train/class1', exist_ok=True)
os.makedirs('/content/synthetic/train/class2', exist_ok=True)

# Generate random images
for i in range(num_images):
    create_random_image(img_size, f'/content/synthetic/train/class1/image_{i}.jpg')
    create_random_image(img_size, f'/content/synthetic/train/class2/image_{i}.jpg')


# In[25]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = '/content/synthetic/train'

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # Use 'categorical' if more than two classes
)


# In[26]:


import tensorflow as tf

# VGG-16 Model
def build_vgg16_model():
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# DenseNet-201 Model
def build_densenet201_model():
    base_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train VGG-16
vgg16_model = build_vgg16_model()
vgg16_history = vgg16_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10
)

# Train DenseNet-201
densenet201_model = build_densenet201_model()
densenet201_history = densenet201_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10
)


# In[27]:


import matplotlib.pyplot as plt

# Plot VGG-16 Training History
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(vgg16_history.history['accuracy'], label='accuracy')
plt.plot(vgg16_history.history['loss'], label='loss')
plt.title('VGG-16 Training History')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()

# Plot DenseNet-201 Training History
plt.subplot(1, 2, 2)
plt.plot(densenet201_history.history['accuracy'], label='accuracy')
plt.plot(densenet201_history.history['loss'], label='loss')
plt.title('DenseNet-201 Training History')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:





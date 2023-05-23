import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models, layers

# Directorio de extracción de los archivos ZIP
extract_dir = '/content/quickdraw'

# cargamos imagenes y etiquetas
def load_images_from_txt(file_path, mapping):
    images = []
    labels = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        image_path, label = line.strip().split('\t')
        image_path = os.path.join('/content/quickdraw/QuickDraw-10', image_path)
        
        if os.path.isfile(image_path) and image_path.endswith('.jpg'):
            image = Image.open(image_path).convert('RGB')
            image = image.resize((64, 64))
            images.append(np.array(image))
            labels.append(int(label))
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels


#aca se cargan los datos QuickDraw-10
data_dir_10 = os.path.join(extract_dir, 'QuickDraw-10')

#se cargan  etiquetas de mapeo  QuickDraw-10
mapping_file_10 = os.path.join(data_dir_10, 'mapping.txt')
with open(mapping_file_10, 'r') as file:
    mapping_10 = dict(line.strip().split('\t') for line in file.readlines())

#Carga de las imágenes y etiquetas de entrenamiento de QuickDraw-10
train_file_10 = os.path.join(data_dir_10, 'train.txt')
train_images_10, train_labels_10 = load_images_from_txt(train_file_10, mapping_10)

# cargar las imágenes y etiquetas de prueba de QuickDraw-10
test_file_10 = os.path.join(data_dir_10, 'test.txt')
test_images_10, test_labels_10 = load_images_from_txt(test_file_10, mapping_10)

#normalizar los valores de píxeles en el rango [0, 1]
train_images_10 = train_images_10 / 255.0
test_images_10 = test_images_10 / 255.0

# MLP
mlp_model = models.Sequential()
mlp_model.add(layers.Flatten(input_shape=(64, 64, 3)))
mlp_model.add(layers.Dense(128, activation='relu'))
mlp_model.add(layers.Dense(64, activation='relu'))
mlp_model.add(layers.Dense(len(mapping_10), activation='softmax'))

#Compilar el modelo MLP
mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

mlp_history = mlp_model.fit(train_images_10, train_labels_10, epochs=10, validation_data=(test_images_10, test_labels_10))

#Evaluar el rendimiento del modelo MLP en los datos de prueba QuickDraw-10
mlp_scores = mlp_model.evaluate(test_images_10, test_labels_10, verbose=0)
print("MLP - Precisión en los datos de prueba QuickDraw-10:", mlp_scores[1])

# Predecir las etiquetas para los datos de prueba QuickDraw-10 utilizando el modelo MLP
mlp_predictions = mlp_model.predict(test_images_10)
mlp_predicted_labels = np.argmax(mlp_predictions, axis=1)

# Mostrar algunos ejemplos de predicciones del modelo MLP
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.imshow(test_images_10[i])
    ax.axis('off')
    predicted_label = list(mapping_10.keys())[mlp_predicted_labels[i]]
    actual_label = list(mapping_10.keys())[test_labels_10[i]]
    ax.set_title(f"Predicted: {predicted_label}\nActual: {actual_label}")
plt.tight_layout()
plt.show()

#CNN
cnn_model = models.Sequential()
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(128, activation='relu'))
cnn_model.add(layers.Dense(len(mapping_10), activation='softmax'))

#compilar el modelo CNN
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn_history = cnn_model.fit(train_images_10, train_labels_10, epochs=10, validation_data=(test_images_10, test_labels_10))

# Evaluar el rendimiento del modelo CNN en los datos de prueba QuickDraw-10
cnn_scores = cnn_model.evaluate(test_images_10, test_labels_10, verbose=0)
print("CNN - Precisión en los datos de prueba QuickDraw-10:", cnn_scores[1])

# Predecir las etiquetas para los datos de prueba QuickDraw-10 utilizando el modelo CNN
cnn_predictions = cnn_model.predict(test_images_10)
cnn_predicted_labels = np.argmax(cnn_predictions, axis=1)

# Mostrar algunos ejemplos de predicciones del modelo CNN
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.imshow(test_images_10[i])
    ax.axis('off')
    predicted_label = list(mapping_10.keys())[cnn_predicted_labels[i]]
    actual_label = list(mapping_10.keys())[test_labels_10[i]]
    ax.set_title(f"Predicted: {predicted_label}\nActual: {actual_label}")
plt.tight_layout()
plt.show()


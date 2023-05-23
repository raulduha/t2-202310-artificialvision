import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models, layers
from zipfile import ZipFile


#archivos zip ruta
extract_dir = '/content/quickdraw/QuickDraw-Animals'

# aca hacemos load de datos
def load_images_from_folder(folder_path, mapping):
    images = []
    labels = []
    
    for label_name in mapping.keys():
        label = mapping[label_name]
        label_folder = os.path.join(folder_path, label_name)
        
        for filename in os.listdir(label_folder):
            if filename.endswith('.jpg'):
                image_path = os.path.join(label_folder, filename)
                image = Image.open(image_path).convert('RGB')
                image = image.resize((64, 64))
                images.append(np.array(image))
                labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels


# cargamos las etiquetas de mapeo
mapping_file = os.path.join(extract_dir, 'mapping.txt')
with open(mapping_file, 'r') as file:
    mapping_lines = file.readlines()

mapping = {}
for line in mapping_lines:
    label_name, label = line.strip().split('\t')
    mapping[label_name] = int(label)

#cargamos las imágenes y etiquetas de entrenamiento
train_zip_file = os.path.join(extract_dir, 'train_images.zip')
with ZipFile(train_zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

train_images_folder = os.path.join(extract_dir, 'train_images')
train_images, train_labels = load_images_from_folder(train_images_folder, mapping)

# cargamos las imágenes y etiquetas de prueba
test_zip_file = os.path.join(extract_dir, 'test_images.zip')
with ZipFile(test_zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

test_images_folder = os.path.join(extract_dir, 'test_images')
test_images, test_labels = load_images_from_folder(test_images_folder, mapping)

# normalizamos los valores de píxeles en el rango [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# MLP
mlp_model = models.Sequential()
mlp_model.add(layers.Flatten(input_shape=(64, 64, 3)))
mlp_model.add(layers.Dense(128, activation='relu'))
mlp_model.add(layers.Dense(64, activation='relu'))
mlp_model.add(layers.Dense(len(mapping), activation='softmax'))

# Compilar el modelo MLP
mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

mlp_history = mlp_model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# evaluamos el rendimiento del modelo MLP en los datos de prueba
mlp_scores = mlp_model.evaluate(test_images, test_labels, verbose=0)
print("MLP - Precisión en los datos de prueba:", mlp_scores[1])

#se predicen las etiquetas para los datos de prueba utilizando el modelo MLP
mlp_predictions = mlp_model.predict(test_images)
mlp_predicted_labels = np.argmax(mlp_predictions, axis=1)

# Mostrar algunos ejemplos de predicciones del modelo MLP
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.imshow(test_images[i])
    ax.axis('off')
    predicted_label = list(mapping.keys())[mlp_predicted_labels[i]]
    actual_label = list(mapping.keys())[test_labels[i]]
    ax.set_title(f"Predicted: {predicted_label}\nActual: {actual_label}")
plt.tight_layout()
plt.show()

# CNN
cnn_model = models.Sequential()
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(128, activation='relu'))
cnn_model.add(layers.Dense(len(mapping), activation='softmax'))

# compilar el modelo CNN
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn_history = cnn_model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluar el rendimiento del modelo CNN en los datos de prueba
cnn_scores = cnn_model.evaluate(test_images, test_labels, verbose=0)
print("CNN - Precisión en los datos de prueba:", cnn_scores[1])

# Predecir las etiquetas para los datos de prueba utilizando el modelo CNN
cnn_predictions = cnn_model.predict(test_images)
cnn_predicted_labels = np.argmax(cnn_predictions, axis=1)

# Mostrar algunos ejemplos de predicciones del modelo CNN
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.imshow(test_images[i])
    ax.axis('off')
    predicted_label = list(mapping.keys())[cnn_predicted_labels[i]]
    actual_label = list(mapping.keys())[test_labels[i]]
    ax.set_title(f"Predicted: {predicted_label}\nActual: {actual_label}")
plt.tight_layout()
plt.show()

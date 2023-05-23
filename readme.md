para QuickDraw-10.py

estos son los pasos de ejecucion

Instrucciones para ejecutar el código
Estas instrucciones te guiarán para ejecutar el código y obtener los resultados del modelo MLP y del modelo CNN en los datos de prueba QuickDraw-10.

Requisitos previos
Antes de ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas de Python:

numpy
matplotlib
Pillow
tensorflow
Puedes instalar las bibliotecas faltantes utilizando el administrador de paquetes pip. Ejecuta el siguiente comando en tu entorno de Python:


pip install numpy matplotlib Pillow tensorflow

Paso 1: Descargar los archivos de QuickDraw-10
Descarga el archivo ZIP de QuickDraw-10 desde el siguiente enlace: enlace al archivo ZIP de QuickDraw-10.

Descomprime el archivo ZIP en una ubicación de tu elección. Asegúrate de recordar la ruta de la carpeta de extracción, ya que la necesitarás en los pasos siguientes.

Paso 2: Configurar la ruta de extracción
En el código, busca la línea que dice:


extract_dir = '/content/quickdraw'
Reemplaza '/content/quickdraw' con la ruta de la carpeta de extracción que obtuviste en el Paso 1.

Paso 3: Cargar los datos de entrenamiento y prueba
Asegúrate de que los archivos train.txt y test.txt estén presentes en la carpeta de extracción de QuickDraw-10. Estos archivos contienen las rutas de las imágenes de entrenamiento y prueba, respectivamente.

Paso 4: Ejecutar el código
Ejecuta el código en tu entorno de Python. Asegúrate de tener suficiente potencia de cómputo para entrenar los modelos, ya que puede llevar algún tiempo dependiendo de tu hardware.

El código entrenará y evaluará dos modelos: un modelo MLP y un modelo CNN. Al finalizar, mostrará algunos ejemplos de predicciones realizadas por cada modelo.

Resultados
Una vez que el código haya terminado de ejecutarse, se mostrará en la salida los resultados de precisión (accuracy) obtenidos por el modelo MLP y el modelo CNN en los datos de prueba QuickDraw-10.

Además, se mostrará una matriz de ejemplos de predicciones del modelo MLP y otra del modelo CNN. Cada matriz mostrará una cuadrícula de imágenes de prueba junto con las etiquetas predichas por los modelos y las etiquetas reales.

para QuickDraw-Animals.py

Código para clasificación de imágenes utilizando MLP y CNN
Este código realiza la clasificación de imágenes de animales utilizando dos modelos de aprendizaje automático: MLP (Perceptrón Multicapa) y CNN (Red Neuronal Convolucional). El código carga un conjunto de imágenes de entrenamiento y prueba, entrena los modelos y evalúa su rendimiento en los datos de prueba.

Requisitos
Python 3.x
Bibliotecas de Python: numpy, matplotlib, Pillow, tensorflow
Puedes instalar las bibliotecas requeridas ejecutando el siguiente comando:
pip install numpy matplotlib Pillow tensorflow


Código para clasificación de imágenes utilizando MLP y CNN
Este código realiza la clasificación de imágenes de animales utilizando dos modelos de aprendizaje automático: MLP (Perceptrón Multicapa) y CNN (Red Neuronal Convolucional). El código carga un conjunto de imágenes de entrenamiento y prueba, entrena los modelos y evalúa su rendimiento en los datos de prueba.

Requisitos
Python 3.x
Bibliotecas de Python: numpy, matplotlib, Pillow, tensorflow
Puedes instalar las bibliotecas requeridas ejecutando el siguiente comando:

Copy code
pip install numpy matplotlib Pillow tensorflow
Instrucciones
Descarga el conjunto de datos QuickDraw-Animals desde este enlace.

Descomprime el archivo zip en la carpeta /content/quickdraw/QuickDraw-Animals.

Abre un entorno de desarrollo Python y ejecuta.

Espera a que finalice la ejecución del script. Los modelos MLP y CNN se entrenarán y evaluarán en los datos de prueba.

Al finalizar, se mostrará la precisión obtenida por cada modelo en los datos de prueba, así como ejemplos de predicciones realizadas por ambos modelos.



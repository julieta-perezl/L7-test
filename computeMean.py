# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:10:11 2024

@author: juli
"""
from astropy.io import fits
import numpy as np
import os



def calculate_adu_matrix(image_path):
    # Abrir la imagen FITS
    hdul = fits.open(image_path)

    # Obtener datos de la imagen
    data = hdul[0].data

    # Calcular la suma de los valores de píxeles en el eje de color (si es una imagen a color)
    adu_matrix = np.sum(data, axis=-1)

    return adu_matrix

def main(image_folder):
    # Obtener lista de archivos de imagen en la carpeta
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.fits')]

    # Verificar si hay imágenes FITS en la carpeta
    if not image_files:
        print("No se encontraron archivos FITS en la carpeta:", image_folder)
        return

    # Obtener dimensiones de la primera imagen para crear la matriz
    first_image_path = os.path.join(image_folder, image_files[0])
    hdul = fits.open(first_image_path)
    height, width = hdul[0].data.shape

    # Crear matriz vacía para almacenar la suma acumulada de ADU de todas las imágenes
    adu_sum_matrix = np.zeros((height, width), dtype=np.uint64)

    # Calcular la suma acumulada de ADU de todas las imágenes
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        adu_matrix = calculate_adu_matrix(image_path)
        adu_sum_matrix += adu_matrix

    # Dividir cada elemento de la matriz por la cantidad de imágenes
    num_images = len(image_files)
    average_adu_matrix = adu_sum_matrix / num_images

    return average_adu_matrix

# Carpeta que contiene los archivos FITS
image_folder = "tu_carpeta_de_imagenes"

# Obtener la matriz promedio de ADU de todas las imágenes en la carpeta
average_adu_matrix = main(image_folder)

# Imprimir la matriz promedio de ADU
print("Matriz promedio de ADU:")
print(average_adu_matrix)

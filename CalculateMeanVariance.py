from astropy.io import fits
import numpy as np
import os
from astropy.nddata import block_reduce
# Directorio que contiene los archivos FITS
#directorio = r'C:\Users\georg\OneDrive\Escritorio\Labo 6\codigos labo 6\im_sintetizada'
directorio = r'C:\Users\administrador\Desktop\L7\Imagenes_con_camara'
#directorio = r'F:\filtros_camara'
#directorio = r'D:\imagenes de javier\fits'


# Contador para llevar la cuenta del número total de imágenes
total_imagenes = 0

datos_imagenes = []

i=0
# Itera sobre todos los archivos FITS en el directorio
for archivo in os.listdir(directorio):
    i+=1
    print(i)
    if archivo.endswith('.fits'):
        ruta_archivo = os.path.join(directorio, archivo)
        with fits.open(ruta_archivo) as hdulist:
            datos_imagen = hdulist[0].data.astype(np.float64)-256 # hay que restar 256 si no son de las matrices reducidas
            datos_imagenes.append(datos_imagen)
            total_imagenes += 1
            if total_imagenes == 1:
                promedio_datos = datos_imagen.copy()  # Inicializar con los datos de la primera imagen
            else:
                promedio_datos += (datos_imagen - promedio_datos) / total_imagenes

# Calcular la varianza del conjunto de datos de todas las imágenes
#varianza=[]
#for i in datos_imagen:
 #   x=(i-promedio_datos)**2
 #   varianza.append(x)
carga=datos_imagen.flatten()
##print(varianza, "lista varianza")
#varianza_datos=np.sum(varianza)/(total_imagenes-1)

varianza_datos = np.sum((datos_imagen - promedio_datos) ** 2 for datos_imagen in datos_imagenes) / (total_imagenes-1)

# Convertir los datos de la varianza a uint16
#varianza_final = varianza_acumulada.astype(np.uint16)

# Guardar la imagen FITS con los datos de la varianza
#nombre_archivo_varianza = 'varianza_imagen.fits'
#fits.writeto(nombre_archivo_varianza, varianza_final, overwrite=True)

#print(f"Se ha creado la imagen de varianza '{nombre_archivo_varianza}' con los datos de varianza del conjunto de {total_imagenes} imágenes.")

print(promedio_datos.shape, "dimension del promedio")
print(varianza_datos.shape, "dimension de la varianza")
print(promedio_datos, "matriz promedio")
print(varianza_datos, "matriz varianza")
hay_negativos = np.any(promedio_datos < 0)

# Imprimir el resultado
if hay_negativos:
    print("Sí, hay valores negativos en promedio_datos.")
else:
    print("No, no hay valores negativos en promedio_datos.")
#-------------------------------
# Guardar la imagen FITS con los datos de la varianza
#nombre_archivo_varianza = 'varianza_imagen.fits'
#fits.writeto(nombre_archivo_varianza, varianza_acumulada, overwrite=True)

# Guardar la imagen FITS con los datos del promedio
#nombre_archivo_promedio = 'promedio_imagen.fits'
#fits.writeto(nombre_archivo_promedio, promedio_datos, overwrite=True)

#print(f"Se han creado las imágenes de promedio '{nombre_archivo_promedio}' y varianza '{nombre_archivo_varianza}'.")

import matplotlib.pyplot as plt

# Leer los datos del archivo FITS de promedio y varianza
#promedio_datos = fits.getdata('promedio_imagen.fits')
#varianza_datos = fits.getdata('varianza_imagen.fits')

# Convertir los datos de la imagen de 2D a 1D para facilitar el trazado
promedio_datos = block_reduce(promedio_datos, (8, 8), np.mean)
varianza_datos = block_reduce(varianza_datos, (8, 8), np.mean)
promedio_flat = promedio_datos.flatten()
varianza_flat = varianza_datos.flatten()
ganancia=varianza_flat/promedio_flat
#------------
'''''
umbral = 1000
elementos_mayor_umbral = varianza_flat[varianza_flat > umbral]
print(elementos_mayor_umbral)
print(len(elementos_mayor_umbral))
print(varianza_flat)
print(len(varianza_flat))
'''''

from scipy.optimize import curve_fit

def modelo_lineal(x, g, b):
    return g * x + b

parametros, covarianza = curve_fit(modelo_lineal, promedio_flat, varianza_flat)
# Obtén los parámetros óptimos
a_optimo = parametros[0]
b_optimo = parametros[1]
hist, bins = np.histogram(ganancia, bins=300)

# Encontrar el centro de cada bin
bin_centers = (bins[1:] + bins[:-1]) / 2

# Definir una función gaussiana
def gaussiana(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2)

# Ajustar la gaussiana al histograma
parametros_optimos_gauss, covarianza_gauss = curve_fit(gaussiana, bin_centers, hist, p0=[1, np.mean(ganancia), np.std(ganancia)])

# Imprimir los parámetros óptimos
print("Parámetros óptimos de la gaussiana:")
print("Amplitud:", parametros_optimos_gauss[0])
print("Media:", parametros_optimos_gauss[1])
print("Desviación estándar:", parametros_optimos_gauss[2])
print("Matriz de covarianza:", covarianza_gauss)





# Imprime el valor del parámetro 'a' óptimo
print("Ganancia por curve_fit:", a_optimo)

# Calcula la varianza ajustada utilizando el modelo lineal y los parámetros óptimos
varianza_ajustada = modelo_lineal(promedio_flat, a_optimo, b_optimo)

print("la ganancia es", np.var(varianza_flat,ddof=1)/np.mean(promedio_flat))
print("matriz de covarianza",covarianza)

#----------------------------------
# Graficar la varianza vs el promedio
plt.figure(1)
#plt.figure(figsize=(8, 6))
plt.scatter(promedio_flat, varianza_flat, s=1, alpha=0.5, label='Datos')
plt.plot(promedio_flat, varianza_ajustada, color='red', label='Ajuste lineal')
#plt.scatter(promedio_flat, ganancia, s=1, alpha=0.5)
#plt.ylim(0, 1000)
plt.xlabel('Esperanza')
plt.ylabel('Varianza')
plt.title('Varianza vs Esperanza')
plt.grid(True)
plt.legend()

# Crear el mapa de colores

plt.figure(2)
plt.hist2d(promedio_flat, varianza_flat, bins=100, cmap='inferno')
#plt.ylim(0, 1000)
plt.xlabel('Esperanza')
plt.ylabel('Varianza')


# Agregar barra de colores
plt.colorbar(label='Intensidad')



# Graficar el histograma de la ganancia
plt.figure(3)
plt.hist(ganancia, bins=300, label='Datos', alpha=0.5, color='b')

# Graficar el ajuste gaussiano
plt.plot(bin_centers, gaussiana(bin_centers, *parametros_optimos_gauss), color='r', label='Ajuste gaussiano')

# Configurar etiquetas y título
plt.xlabel('Ganancia')
plt.ylabel('Frecuencia')
plt.title('Histograma de ganancia con ajuste gaussiano')
plt.grid(True)
plt.legend()
#plt.hist(ganancia,bins=300)

#plt.xlim(0,0.5)

#plt.legend()

plt.figure(4)
plt.hist(carga,bins=50)
plt.title('Histograma de carga')
plt.ylabel('Frecuencia')
plt.xlabel('Carga')
plt.xlim(0,800)
plt.grid(True)



# Calcula los residuos entre los valores de varianza y los valores predichos por la recta ajustada
residuos = varianza_flat - varianza_ajustada

# Establece un umbral para los residuos
umbral_residuos = 10  # Por ejemplo, ajusta este valor según lo necesites

# Encuentra las coordenadas de los píxeles que cumplen con el criterio del umbral
puntos_anomalos = np.where(np.abs(residuos) > umbral_residuos)[0]

# Grafica la esperanza de esos píxeles en función del número de imagen
plt.figure(5)
plt.plot(range(len(datos_imagenes)), [datos_imagenes[i].flatten()[puntos_anomalos] for i in range(len(datos_imagenes))])
plt.xlabel('Número de imagen')
plt.ylabel('Esperanza')
plt.title('Esperanza de píxeles anómalos en función del número de imagen')
plt.grid(True)



plt.show()




''''
esperanza_deseada = 500
margen = 1  # Puedes ajustar el tamaño del margen según tus necesidades

# Definir el rango de esperanza
esperanza_minima = esperanza_deseada - margen
esperanza_maxima = esperanza_deseada + margen

# Lista para almacenar las varianzas dentro del rango de esperanza
varianzas_esperanza_deseada = []

# Iterar sobre los pares de esperanza y varianza
for esperanza, varianza in zip(promedio_flat, varianza_flat):
    # Verificar si la esperanza está dentro del rango deseado
    if esperanza_minima <= esperanza <= esperanza_maxima:
        # Agregar la varianza a la lista
        varianzas_esperanza_deseada.append(varianza)

# Imprimir la lista de varianzas dentro del rango de esperanza deseado
print("Varianzas dentro del rango de esperanza", esperanza_minima, "-", esperanza_maxima, ":", np.var(varianzas_esperanza_deseada,ddof=1))
'''''

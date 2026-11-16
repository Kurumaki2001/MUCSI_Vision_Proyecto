DATASET SINTETICO: PUNTOS PARA YOLO

Total: 100 imagenes JPG de 640x640.
Entrenamiento: 80 imagenes (images/train) y 80 etiquetas (labels/train).
Validacion: 20 imagenes (images/val) y 20 etiquetas (labels/val).
Clase 0: punto.

Variaciones incluidas:
- Diferentes colores (negro, grises, rojo, naranja, amarillo, verde, azul,
  cian, violeta, rosa y marron).
- Diferentes niveles de luminosidad.
- Diferentes niveles de nitidez, desde bordes definidos hasta desenfoque.
- Diferentes tamanos y posiciones.

Las etiquetas usan el formato YOLO: clase, centro_x, centro_y, ancho, alto,
con coordenadas normalizadas entre 0 y 1.

Nota: son datos sinteticos. Conviene probar el modelo con imagenes reales
del entorno donde se vaya a utilizar.

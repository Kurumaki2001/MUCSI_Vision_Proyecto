# Proyecto: emparejar puntos y números (enfoque de visión artificial)

## Contexto
Proyecto universitario de visión por computador, en Python. Las imágenes tienen fondo blanco, con puntos y números impresos. Cada punto tiene asociado un número, que se identifica por la cercanía. Hay que detectar los puntos, leer los números y emparejar cada punto con su número.

El problema se resuelve con dos enfoques que después compararé: visión clásica y visión artificial. Ahora solo quiero el de visión artificial. El clásico lo haré más adelante y reutilizará los módulos comunes (formato de resultados, emparejamiento, evaluación y visualización), así que esos módulos no deben depender de nada de la parte de IA.

## Enfoque
1. Detección con Ultralytics YOLO26, partiendo de `yolo26n.pt` y reentrenado con dos clases: 0 = `punto` y 1 = `numero` (una caja por número completo, no por dígito).
2. Lectura con EasyOCR limitado a dígitos (`allowlist="0123456789"`). Antes de leer, borra los puntos detectados pintándolos de blanco con un pequeño margen, para que el OCR no los confunda con ceros ni los pegue a los números. Implementa dos modos, elegibles en la configuración, para poder compararlos:
   - `pagina`: EasyOCR sobre la imagen entera. Puede fusionar números cercanos en una sola caja, así que ajusta sus parámetros de fusión y descarta las lecturas que no sean solo dígitos.
   - `recortes`: EasyOCR lee cada caja `numero` de YOLO como una sola línea, sin usar su propio detector (recorta con margen y amplía si es pequeña). Si una caja no se puede leer, mantenla en el emparejamiento con valor desconocido, para que su punto no se quede con un número ajeno.
   En los dos modos, descarta las lecturas con confianza baja (umbral configurable).
3. Emparejamiento con el algoritmo húngaro (`scipy.optimize.linear_sum_assignment`), usando la distancia del centro de cada punto al rectángulo de cada número, no entre centros. Fija una distancia máxima relativa al tamaño de los números (por ejemplo, 3 veces la altura mediana de sus cajas): por encima no hay pareja. Devuelve también los puntos y números sin pareja.
4. Si los números forman una secuencia 1..N, avisa de los que faltan o se repiten (solo como diagnóstico).

## Datos sintéticos (generador con Pillow)
- Fondo blanco con puntos rellenos y, junto a cada uno, su número a una distancia y en una dirección aleatorias. Números del 1 al N, con N aleatorio dentro de un rango. Sin solapes, salvo un porcentaje configurable de casos difíciles (un número que toca su punto, números muy juntos).
- Variedad: varias fuentes, tamaños y grosores; ruido, desenfoque, rotación ligera, compresión JPEG e iluminación irregular suave. Si rotas, recalcula bien las cajas.
- Salidas: imágenes, etiquetas en formato YOLO, `data.yaml` para Ultralytics y un JSON de anotación por imagen con toda la verdad (puntos, números con su valor y su caja, y parejas). Divisiones train/val/test (2000/300/200 por defecto) con semilla fija.
- Todos los parámetros (tamaño de imagen, radio de los puntos, tamaño de letra, distancia punto-número, cantidad de puntos) en un fichero de configuración. Si hay imágenes en `datos/reales/`, míralas y propón valores para que las sintéticas se parezcan.
- Guarda algunas imágenes con las cajas dibujadas para comprobar a simple vista que las etiquetas cuadran.

## Entrenamiento
- Script con `imgsz` alto (1024 por defecto, porque los objetos son pequeños), parada temprana, semilla fija y volteo horizontal desactivado (`fliplr=0`), porque los números nunca aparecen en espejo.
- Que use la GPU si la hay. Si no hay GPU local, entrenaré en Google Colab: añade un notebook o instrucciones paso a paso para entrenar allí y descargar los pesos.
- Conserva las métricas y gráficas que genera Ultralytics (mAP, matriz de confusión), porque las usaré en la memoria.

## Inferencia
- Script que procese una imagen o una carpeta y guarde, para cada imagen:
  - Un JSON de resultados con un formato común a los dos enfoques: puntos (id, x, y, confianza), números (id, valor o null, caja [x0, y0, x1, y1], confianza), parejas (id del punto, id del número, distancia), puntos y números sin pareja, y tiempo total en ms con su desglose por etapas. Las anotaciones usan el mismo esquema, sin confianzas ni tiempos.
  - Una imagen anotada con los puntos, las cajas de los números con el valor leído, una línea de cada punto a su número y los elementos sin pareja en rojo. Como opción, la línea que une los puntos en orden numérico.
- Si mis imágenes reales son mucho más grandes que `imgsz`, plantea trocearlas en teselas con solape en lugar de reducirlas.

## Evaluación (común a los dos enfoques)
- Script que compare los JSON de resultados con los de anotación y calcule:
  - Puntos: precisión, recall y F1. Un punto predicho acierta si cae a menos de una tolerancia configurable de uno real, con asignación uno a uno.
  - Números: % leídos correctamente (solo si la anotación incluye sus cajas).
  - Parejas: % de puntos reales cuyo número asignado tiene el valor correcto.
  - % de imágenes perfectas y tiempo medio por imagen.
- En una anotación real mínima, los números pueden ir sin caja. Incluye un script sencillo con OpenCV para anotar imágenes reales haciendo clic en cada punto y escribiendo su número.
- Opción de robustez: evaluar sobre copias del test con perturbaciones (ruido, desenfoque, rotación, baja resolución, sombras) y sacar una tabla de resultados por perturbación.

## Estructura sugerida
- `config.yaml`
- `comun/`: `formato.py`, `emparejar.py`, `evaluar.py`, `dibujar.py` (compartidos con el enfoque clásico)
- `datos/`: `generar.py` y la carpeta `reales/`
- `ia/`: `entrenar.py`, `pipeline.py`
- `herramientas/anotar.py`
- `tests/`, `README.md`, `requirements.txt`

## Forma de trabajar
- Antes de escribir código, preséntame un plan breve y espera mi visto bueno.
- Trabaja por fases y, al terminar cada una, resume qué has hecho y cómo probarlo antes de seguir: (1) estructura, configuración y entorno virtual; (2) generador; (3) formato, emparejamiento y evaluación; (4) entrenamiento, con una prueba rápida de pocas imágenes y 1–2 épocas solo para comprobar que funciona; (5) inferencia con los dos modos de OCR; (6) README.
- Tests con pytest, al menos para el emparejamiento (incluidos casos ambiguos), las etiquetas YOLO (coordenadas normalizadas y dentro de la imagen) y las métricas (una predicción perfecta debe dar el 100 %).
- Código claro y comentado en español: tengo que entenderlo y defenderlo. Prioriza la claridad sobre los trucos y no añadas dependencias innecesarias. Comprueba que EasyOCR y Ultralytics conviven en el mismo entorno (las dos instalan OpenCV) y que el script de anotación puede abrir ventanas.
- El README debe explicar cómo ejecutar cada paso y las decisiones de diseño: por qué YOLO + OCR, por qué dos modos de OCR, por qué borrar los puntos antes del OCR, por qué el algoritmo húngaro con distancia punto-rectángulo y por qué `fliplr=0`. Me servirá para la memoria.
- No inventes resultados: cualquier cifra tiene que salir de ejecutar el código.
- Si algo no está claro, pregúntame.
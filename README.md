# Emparejar puntos y números en fichas de "unir los puntos"

Proyecto de la asignatura de Visión Artificial (Máster en IA, Universidad de Deusto).

## Qué se quiere hacer

Una ficha de "unir los puntos" es una hoja con puntos impresos y, junto a cada uno, un número (1, 2, 3...). Si se unen en orden, aparece un dibujo. El objetivo es, a partir de una **foto cenital de la ficha**:

1. **detectar** todos los puntos,
2. **leer** todos los números,
3. **emparejar** cada punto con su número, el que tiene más cerca,
4. y, si en la foto aparece la hoja de marcadores ArUco de clase, dar la **posición real de cada punto en milímetros** sobre la mesa. Con eso un robot podría recorrer los puntos en orden.

El problema se resuelve con dos enfoques que se compararán al final, con el mismo formato de resultados y la misma evaluación:

- **`ArtificialVision/`**, visión artificial: un detector YOLO26 reentrenado localiza puntos y números, y EasyOCR lee los números. **Está hecho.**
- **`ClassicVision/`**, visión clásica: solo OpenCV y reglas (umbralizado, contornos, plantillas de dígitos), sin entrenar nada. **En desarrollo.**

El enunciado completo está en [Proyecto.md](Proyecto.md).

## Cómo se ha hecho, en una imagen

```
                 data/generate.py                      ArtificialVision/prepare_yolo.py
config.yaml ──►  fotos sintéticas + JSON (verdad) ──►  teselas de 1024 px + etiquetas YOLO
                         │                                          │
                         │                              ArtificialVision/train.py
                         │                                          ▼
                         │                        models/yolo26n_dots_numbers.pt
                         │                                          │
foto (sintética o real) ─┼──────────────►  ArtificialVision/pipeline.py
                         │                   1. YOLO por teselas: cajas "dot" y "number"
                         │                   2. ArUco: hoja de referencia (opcional)
                         │                   3. borrar los puntos
                         │                   4. EasyOCR, solo dígitos (modo page o crops)
                         │                   5. emparejar (algoritmo húngaro)
                         │                   6. pasar a mm con la homografía de ArUco
                         │                                          │
                         │                      <foto>.json, <foto>.jpg, <foto>_mm.csv
                         ▼                                          ▼
                    common/evaluation.py: compara resultados con la verdad → métricas
```

Por ahora no hay fotos reales anotadas, así que el modelo se entrena **solo con fotos sintéticas**. Un generador simula fotos de una ficha A4 hechas con una cámara cenital, y como sabe dónde ha dibujado cada cosa, la anotación es exacta. Las fotos reales se sacan de vídeos grabados con la cámara del proyecto y se etiquetan con SAM (`tools/annotate_sam.py`), como en las prácticas de clase. Servirán para evaluar el sistema en condiciones reales y, después, para afinar el entrenamiento.

## Qué hay en el repositorio

```
MUCSI_Vision_Proyecto/
├── Proyecto.md              Enunciado del proyecto.
├── README.md                Este fichero.
├── config.yaml              TODOS los parámetros (generador, YOLO, entrenamiento, inferencia, ArUco,
│                            emparejamiento y evaluación), comentados. Ningún script tiene valores fijos.
├── requirements.txt         Dependencias con versión (PyTorch con CUDA lo instala el script de abajo).
├── setup_environment.ps1    Crea el entorno virtual .venv e instala todo (Windows).
├── pytest.ini               Configuración de los tests.
│
├── common/                  Módulos compartidos por los dos enfoques. No importan nada de
│   │                        ArtificialVision/ ni de ClassicVision/.
│   ├── config.py            Lee config.yaml y resuelve rutas relativas a la carpeta del proyecto.
│   ├── json_format.py       Formato JSON común de anotaciones y resultados: crear, validar, leer, guardar.
│   ├── matching.py          Emparejamiento punto-número (algoritmo húngaro) y diagnóstico de la secuencia 1..N.
│   ├── evaluation.py        Métricas comparando resultados con anotaciones (ejecutable).
│   ├── drawing.py           Dibuja una anotación o un resultado sobre la foto.
│   ├── images.py            Lee y escribe imágenes con OpenCV, también con tildes en la ruta.
│   ├── aruco_coords.py      Coordenadas en mm con la hoja de marcadores ArUco, para los dos enfoques
│   │                        (ejecutable por separado).
│   ├── aruco_capture.py     Captura fotos de la hoja ArUco con la cámara, para calibrar la lente.
│   └── aruco_calibrate.py   Calibra la distorsión de la lente con esas fotos.
│
├── aruco_sheet/             Hoja de marcadores (PDF para imprimir y JSON con sus medidas reales),
│                            calibración de la cámara de clase y las fotos usadas para calibrarla.
│
├── data/
│   ├── generate.py          Generador de fotos sintéticas y su anotación (ejecutable).
│   ├── real/                Fichas reales: videos/ (no está en git), images/ (fotogramas vXX_NNNNN.jpg)
│   │                        y annotations/ (un JSON por fotograma).
│   ├── synthetic/           (no está en git) Dataset sintético: train/val/test, con imágenes y JSON.
│   └── synthetic_quick/     (no está en git) Dataset mini de la prueba rápida.
│
├── ArtificialVision/        Enfoque de visión artificial.
│   ├── prepare_yolo.py      Convierte el dataset común en un dataset YOLO troceado en teselas (ejecutable).
│   ├── train.py             Entrena YOLO26n y guarda métricas, gráficas y modelo (ejecutable).
│   ├── pipeline.py          Inferencia completa sobre una foto o una carpeta (ejecutable).
│   ├── models/              yolo26n_dots_numbers.pt: el modelo entrenado (yolo26n.pt, el preentrenado,
│   │                        se descarga solo y no está en git).
│   ├── results/training/    Métricas y gráficas del entrenamiento (curvas, matrices de confusión,
│   │                        summary.json). Sirven para la memoria.
│   ├── results/inference/   (no está en git) Salidas del pipeline y sus evaluation.json.
│   └── yolo_dataset/        (no está en git) Teselas y etiquetas YOLO.
│
├── ClassicVision/           Enfoque de visión clásica (en desarrollo).
│   ├── preprocess.py        Separa la hoja de la mesa y la tinta del papel.
│   ├── dots.py              Detecta los puntos por su forma (redondos, rellenos, de tamaño típico).
│   ├── digits.py            Agrupa la tinta restante en números y reconoce cada dígito con plantillas.
│   ├── pipeline.py          Inferencia completa sobre una foto o una carpeta (ejecutable).
│   └── results/inference/   (no está en git) Salidas del pipeline.
│
├── tools/
│   ├── extract_frames.py    Saca fotogramas de los vídeos de las fichas reales (adaptado del material de clase).
│   ├── annotate_sam.py      Etiquetado de fotos reales con SAM 2.1: cajas de puntos y números, valores y
│   │                        parejas (adaptado del material de clase). Con --prelabel, el modelo propone.
│   ├── annotate.py          Anotación mínima de fotos reales: solo el centro de cada punto y su número.
│   └── check_environment.py Comprueba la GPU, que OpenCV abre ventanas y que EasyOCR y Ultralytics conviven.
│
└── tests/                   Tests con pytest de cada módulo.
```

Lo que está marcado como "no está en git" se regenera **idéntico** con los comandos de abajo, porque todo usa la semilla de `config.yaml`.

Convención del código: los nombres (variables, funciones, ficheros, claves del YAML y del JSON) van en inglés, y los comentarios, docstrings y mensajes por consola en español. Las imágenes se tratan siempre con OpenCV.

## Instalación

Requiere Windows, Python 3.12 y, para entrenar en un tiempo razonable, una GPU NVIDIA. Desde la carpeta del proyecto:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1          # con GPU (PyTorch CUDA cu132)
powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1 -NoGpu   # solo CPU
```

El script:

1. crea `.venv`,
2. instala PyTorch con CUDA y `requirements.txt` (Ultralytics 8.4.215, EasyOCR 1.7.2, OpenCV 5),
3. **sustituye `opencv-python-headless`**, que instala EasyOCR y no puede abrir ventanas, por `opencv-python`,
4. y termina ejecutando `tools/check_environment.py`.

Esa comprobación se puede repetir en cualquier momento:

```powershell
.venv\Scripts\python.exe tools\check_environment.py          # GPU, convivencia de librerías, ventanas
.venv\Scripts\python.exe tools\check_environment.py --ocr    # además prueba EasyOCR (descarga sus modelos)
```

En la consola de Windows conviene fijar `$env:PYTHONIOENCODING = "utf-8"` para que los mensajes con tildes se vean bien.

## Cómo ejecutar cada paso

Todos los comandos se lanzan desde la carpeta del proyecto con el Python del entorno (`.venv\Scripts\python.exe`, abreviado aquí como `python`). Los cuatro primeros pasos admiten `--quick`, que usa un dataset mini (16/4/4 fotos) y 2 épocas para comprobar en pocos minutos que todo funciona.

### 1. Generar los datos sintéticos

```powershell
python -m data.generate            # 2000 fotos de train, 300 de val y 200 de test
```

Cada foto simula una ficha A4 fotografiada desde arriba:

- **Resolución:** de 4 a 8 px/mm, lo que da una cámara Full HD o 4K.
- **Puntos:** de 15 a 120, colocados por el contorno de una figura o al azar.
- **Números:** en fuentes TrueType variadas.
- **Casos difíciles:** números que tocan su punto o muy pegados a los vecinos.
- **Distractores sin etiquetar:** título, campos "Nombre:", letras sueltas y circunferencias huecas.
- **Efectos de cámara:** giro, perspectiva, iluminación irregular, sombras, desenfoque, ruido y JPEG.

Salida en `data/synthetic/{train,val,test}/`: `images/photo_XXXXX.jpg` y `annotations/photo_XXXXX.json`. También se guardan `samples/`, con algunas fotos con las cajas dibujadas para revisarlas a ojo, y `config_used.yaml`. **El JSON es la única fuente de verdad**: todo lo demás sale de él.

### 2. Preparar el dataset de YOLO

```powershell
python -m ArtificialVision.prepare_yolo
```

Trocea cada foto en teselas de 1024 × 1024 px con 160 px de solape y escribe sus etiquetas YOLO (clase 0 = `dot`, clase 1 = `number`, una caja por número completo, no por dígito). Con el dataset por defecto salen 10 984 teselas de train, 1683 de val y 1077 de test. En `ArtificialVision/yolo_dataset/samples/` se guardan teselas con las etiquetas dibujadas a partir de los `.txt`, para comprobar que cuadran.

### 3. Entrenar

```powershell
python -m ArtificialVision.train
```

Parte de `yolo26n.pt`, preentrenado en COCO, y lo reentrena en la GPU con `imgsz=1024`, semilla fija, parada temprana (paciencia 20) y un límite de 0,9 horas. Al terminar:

- evalúa el mejor modelo en validación y en test en precisión completa (FP32),
- guarda las gráficas y `summary.json` en `ArtificialVision/results/training/yolo26n_dots_numbers/`,
- copia el modelo a `ArtificialVision/models/yolo26n_dots_numbers.pt`.

El modelo entrenado ya está en el repositorio, así que este paso solo hace falta para reentrenar.

### 4. Inferencia

```powershell
python -m ArtificialVision.pipeline data/real                               # una carpeta, modo de config.yaml
python -m ArtificialVision.pipeline foto.jpg --mode page --join             # una foto, modo page, uniendo en orden
python -m ArtificialVision.pipeline data/synthetic/test/images --mode crops --output ArtificialVision/results/inference/synthetic_test_crops
```

Por cada foto se escriben tres ficheros:

- `<foto>.json`: resultados en el formato común.
- `<foto>.jpg`: dibujo de los puntos, las cajas con el valor leído y una línea de cada punto a su número. Lo que queda sin pareja sale en rojo; con `--join` también se dibuja el recorrido 1 → N.
- `<foto>_mm.csv`: solo si aparece la hoja ArUco. Contiene número, x_mm e y_mm en orden numérico.

Por consola sale una línea por foto con el número de puntos, números y parejas, el tiempo, los números que faltan o se repiten en la secuencia 1..N y el error de la referencia ArUco.

Si no se indica `--output`, los resultados van a `ArtificialVision/results/inference/<modo>/`.

### 5. Evaluar

```powershell
python -m common.evaluation --annotations data/synthetic/test/annotations --results ArtificialVision/results/inference/synthetic_test_crops ArtificialVision/results/inference/synthetic_test_page
```

Saca una tabla con una fila por carpeta de resultados y guarda en cada una `evaluation.json`, con el detalle por foto. Admitir varias carpetas sirve para comparar modos, enfoques o, en la prueba de robustez, perturbaciones. Las métricas son:

- **Puntos:** precisión, recall y F1. Un punto predicho acierta si cae a menos de 8 px de uno real, con asignación uno a uno.
- **Números leídos:** % de números reales con una caja predicha que se solapa con la suya y el mismo valor. Solo se calcula si la anotación tiene las cajas.
- **Parejas correctas:** % de puntos reales a los que el sistema ha asignado un número con el valor correcto. Es la métrica que importa al final, y no necesita cajas.
- **Imágenes perfectas:** % de fotos sin puntos perdidos ni sobrantes y con todas las parejas bien.
- **Tiempo medio** por foto.

### 6. Fotos reales: de los vídeos a las anotaciones

Es el flujo de las prácticas 2.1-2.3 de clase, adaptado a este proyecto: se graban vídeos de las fichas con la cámara del proyecto, se sacan fotogramas y se etiquetan con SAM.

**Cámara del proyecto:** "USB Camera", como mucho 1920×1080 (16:9) a 30 fps. También da 640×480 (la que da si no se le pide otra), 800×600 y 1280×720. Hay que grabar a 1920×1080. Para que el detector vea los puntos como al entrenar, conviene encuadrar de modo que salgan al menos 4 px/mm; se mide con los marcadores ArUco, de 80 mm de lado.

1. **Sacar fotogramas.** Los vídeos se copian a `data/real/videos/`, que no se sube a git porque pesa mucho.

   ```powershell
   python -m tools.extract_frames                        # todos los vídeos de data/real/videos
   python -m tools.extract_frames --min-sharpness 50     # descartando fotogramas movidos
   ```

   Guarda una foto de cada 15 fotogramas en `data/real/images/` con el nombre `vXX_NNNNN.jpg` (vídeo XX, fotograma NNNNN). En `videos.txt` queda qué vídeo es cada `vXX`. Al añadir vídeos otro día, los antiguos conservan su número. Así, al repartir en entrenamiento, validación y test, se puede separar por vídeo, como en clase: los fotogramas de un mismo vídeo son casi iguales y, si cayeran a los dos lados, la validación saldría optimista.

2. **Etiquetar con SAM.**

   ```powershell
   python -m tools.annotate_sam                # empieza en la primera foto sin anotar
   python -m tools.annotate_sam --prelabel     # el modelo actual propone las parejas y solo se corrigen
   ```

   Se etiqueta por parejas:
   - clic en el punto: SAM propone su caja; ESPACIO la fija,
   - clic en su número (o arrastrar una caja alrededor, mejor con números de varias cifras): SAM propone su caja,
   - se teclea el valor,
   - Intro guarda la pareja.
   
   El clic derecho marca "esto no", para quitar sombras o el número de al lado. `e` y `x` editan y borran la pareja que está bajo el ratón, y `z` deshace. La rueda del ratón hace zoom y el botón central mueve la foto. `a`/`d` cambian de foto y `q` sale; los dos guardan antes. La lista completa de controles está al principio de [tools/annotate_sam.py](tools/annotate_sam.py).

   Se guarda `data/real/annotations/<foto>.json` en el formato común, **con las cajas de puntos y números**. Así las fotos reales sirven para evaluarlo todo, incluida la lectura de números, y para entrenar YOLO. Los pesos de SAM 2.1 (unos 160 MB) se descargan solos la primera vez en `ArtificialVision/models/`.

   Detalles:
   - SAM no recibe la foto entera, sino un recorte de 256 px alrededor del clic. SAM amplía su entrada a 1024 px, así que el punto se ve unas 4 veces más grande. En una foto sintética reducida a 2,7 px/mm, la caja de SAM coincidió con la real (IoU ≥ 0,7) en el 100 % de 20 puntos y el 95 % de 20 números. Con la foto entera, en el 90 % y el 65 %.
   - Si SAM devuelve una caja absurda (más de 60 px para un punto o 160 px para un número), se descarta. Si se había dibujado una caja, se usa esa.
   - Con `--prelabel`, el JSON queda marcado con `"prelabel": true`. **Las fotos de test deben etiquetarse sin `--prelabel`.** Si no, se evaluaría el modelo con etiquetas que él mismo propuso y el resultado saldría sesgado a su favor.

`tools/annotate.py` sigue disponible para una anotación mínima y más rápida: un clic en cada punto y su valor, sin cajas. Sirve para evaluar puntos y parejas, pero no la lectura de números ni para entrenar.

```powershell
python -m tools.annotate data/real/images     # anotación mínima, en data/real/images/annotations/
```

### 7. Coordenadas reales con ArUco

La referencia es la hoja de calibración de clase, [arucos_calibracion_A4.pdf](aruco_sheet/arucos_calibracion_A4.pdf): un A4 con 4 marcadores `DICT_4X4_50` (IDs 0-3) de 80 mm. Tiene que estar **en el mismo plano que la ficha**, es decir, encima de la mesa. El pipeline la usa automáticamente (`aruco.enabled` en `config.yaml`). También se puede probar por separado:

```powershell
python -m common.aruco_coords foto.png                    # marcadores encontrados y error en mm
python -m common.aruco_coords foto.png --point 640 480    # pasa un píxel a mm
python -m common.aruco_coords carpeta/ --validate         # validación cruzada
```

Para recalibrar la lente si cambia la cámara, la resolución, el enfoque o mucho la iluminación:

```powershell
python -m common.aruco_capture       # a 1920x1080; ESPACIO o s guarda una foto, a = automático, q o ESC sale
python -m common.aruco_calibrate     # calcula la distorsión y la guarda en calibracion_camara.json
```

La calibración actual (`calibracion_camara.json`) es de la cámara de clase a 1280×960 (4:3). Con la cámara del proyecto (16:9) el código no la aplica, así que **hay que recalibrar** con los dos comandos anteriores.

### Tests

```powershell
python -m pytest
```

Cubren:

- el emparejamiento, incluidos casos ambiguos,
- que las etiquetas YOLO estén normalizadas, dentro de la imagen y coherentes con el JSON,
- las métricas (una predicción perfecta da el 100 % y cada tipo de error se nota donde debe),
- el generador, las herramientas de anotación y de etiquetado con SAM (sin abrir ventanas ni cargar SAM), el extractor de fotogramas, el pipeline sobre fotos generadas y las coordenadas ArUco.

## Formato JSON común

Anotaciones y resultados usan el mismo esquema, definido y validado en [common/json_format.py](common/json_format.py). Las coordenadas van en píxeles de la foto y las cajas como `[x0, y0, x1, y1]`:

```json
{
  "image": "photo_00000.jpg", "width": 1600, "height": 2133,
  "dots":    [{"id": 0, "x": 812.4, "y": 300.1, "box": [806.0, 293.7, 818.8, 306.5]}],
  "numbers": [{"id": 0, "value": 1, "box": [822.0, 290.2, 834.5, 303.0]}],
  "pairs":   [{"dot": 0, "number": 0, "distance": 10.2}],
  "unpaired_dots": [], "unpaired_numbers": [],
  "metadata": {}
}
```

Los resultados añaden `confidence` a cada punto y número, y `times_ms` con el total y el desglose por etapas. Un número que no se ha podido leer lleva `"value": null`. Si se encontró la hoja ArUco, cada punto lleva también `x_mm` e `y_mm`. Gracias a este formato, el enfoque clásico podrá usar la misma evaluación y el mismo dibujo sin tocar nada.

## Decisiones de diseño

### Por qué YOLO + OCR

Son dos problemas distintos y cada herramienta resuelve uno:

- **Localizar** objetos pequeños entre distractores (títulos, letras sueltas, circunferencias, sombras) es lo que hace bien un detector entrenado. YOLO aprende qué es un punto y qué es un número de este tipo de fichas, y da una caja por número completo, sea "7" o "118".
- **Leer** el valor de esa caja es trabajo de un OCR. Entrenar YOLO con una clase por valor (del 1 al 120) necesitaría muchísimos más ejemplos de cada uno.

EasyOCR se limita a dígitos (`allowlist="0123456789"`) para que no confunda un "1" con una "l" ni un "0" con una "O".

### Por qué dos modos de OCR

Para compararlos, porque cada uno falla de forma distinta:

- **`page`:** EasyOCR busca el texto en toda la foto con su propio detector. No depende de las cajas de YOLO, pero puede fusionar dos números cercanos en una sola caja ("12 13" → "1213"). Por eso se bajan sus umbrales de fusión y de tamaño mínimo (`page_params`, ajustados con fotos de validación: con los valores por defecto se perdían muchos números pequeños) y se descartan las lecturas que no son solo dígitos.
- **`crops`:** EasyOCR lee cada caja `number` de YOLO como una sola línea, recortada con margen y ampliada si es pequeña, sin usar su detector. Nunca fusiona números, porque YOLO ya los separa. Si una caja no se puede leer, el número se mantiene en el emparejamiento con valor desconocido: así su punto no se queda con el número de un vecino.

En los dos modos se descartan las lecturas con confianza menor que `ocr_min_confidence`. Las cifras de la comparación están en [Resultados](#resultados); `crops` es el modo por defecto.

### Por qué borrar los puntos antes del OCR

Un punto relleno junto a un número se parece a un "0" o a un punto decimal, así que el OCR puede leerlo como un dígito más o pegarlo al número. Antes de leer, cada punto detectado se tapa con un círculo un 30 % mayor.

El enunciado propone taparlo de blanco, pero el círculo se pinta **del color del papel que lo rodea** (la mediana de un anillo alrededor). Con papel gris por una sombra, un círculo blanco dejaba un borde marcado que el OCR leía como un trazo más: un "22" pegado a su punto se leía "222".

### Por qué el algoritmo húngaro con distancia punto-rectángulo

- **Algoritmo húngaro** (`scipy.optimize.linear_sum_assignment`): elige, de todas las formas de emparejar sin repetir puntos ni números, la de menor distancia total. Un emparejamiento voraz (cada punto con su número más cercano) falla cuando el número más cercano a un punto es en realidad el de su vecino. El húngaro lo resuelve porque mira todas las parejas a la vez.
- **Distancia del centro del punto al rectángulo del número**, no entre centros: el centro de un "118" largo puede quedar más lejos de su punto que el centro de un "4" vecino, aunque su borde esté pegado al punto.
- **Distancia máxima:** 3 veces la altura mediana de los números de la foto. Por encima no hay pareja, y el punto o el número quedan como "sin pareja". Al ser relativa, funciona igual con cualquier resolución.

Además, si los números forman una secuencia 1..N, se avisa de los que faltan o se repiten. Es solo un diagnóstico y no cambia el emparejamiento.

### Por qué `fliplr=0`

Ultralytics voltea horizontalmente la mitad de las imágenes de entrenamiento por defecto. Aquí no tiene sentido: los números **nunca aparecen en espejo** en una foto real, y un "2" o un "5" volteados son formas que no existen. Se desactiva para no enseñar al modelo ejemplos imposibles. Por el mismo motivo tampoco se añade giro (`degrees=0`): el generador ya gira la hoja y recalcula las cajas con exactitud, mientras que el giro de Ultralytics agranda las cajas.

### Otras decisiones

- **Teselas en lugar de reducir la foto:** Ultralytics reduce cada imagen hasta que su lado mayor mide `imgsz`. Una foto de 2900 px se quedaría a un tercio y los puntos, de pocos píxeles, casi desaparecerían. Por eso la foto se trocea en teselas de 1024 px, tanto al entrenar como en la inferencia. El solape de 160 px es mayor que cualquier objeto, y cada objeto se asigna a una sola tesela, así que ninguno sale dos veces.
- **Fotos más grandes que `max_side` (2900 px):** se reducen a ese tamaño antes de trocearlas, para quedar en la resolución con la que se entrenó (4-8 px/mm).
- **Imagen cenital de un A4 en lugar de una "imagen con fondo blanco":** las fotos reales son de una ficha sobre una mesa, con perspectiva, sombras y distractores. El generador lo imita para que el modelo funcione con ellas.
- **OpenCV en lugar de Pillow en el generador:** el enunciado sugiere Pillow, pero todo el proyecto usa OpenCV para trabajar con una sola librería. Las fuentes TrueType se dibujan con `cv2.FontFace`, de OpenCV 5.
- **Números en la evaluación por intersección / área de la caja menor, no por IoU:** las cajas de EasyOCR salen más holgadas que las reales, y con IoU lecturas correctas contaban como fallos. Una caja que fusiona dos números sigue fallando, porque su valor no coincide.
- **Métricas finales en FP32:** durante el entrenamiento Ultralytics valida en media precisión (FP16), que redondea las coordenadas a pasos de 0,5 px. Con objetos tan pequeños eso baja el mAP50-95. Por eso las curvas de `results.png` quedan por debajo de las cifras de `summary.json`, que se recalculan en FP32. `summary.json` guarda también la validación en FP16 para comparar.
- **ArUco para pasar a milímetros:** entre el plano de la mesa y su foto hay una homografía, sea cual sea la perspectiva. Conociendo la posición real de las 16 esquinas de los 4 marcadores y detectándolas en la foto, se calcula esa homografía (`cv2.findHomography`) y se aplica a cada punto. Antes se refinan las esquinas y se corrige la distorsión de la lente. Lo que YOLO detecte dentro de un marcador se descarta, porque sus cuadrados negros pueden parecer puntos.
- **Entrenamiento en local:** el enunciado preveía Google Colab por si no había GPU, pero se ha entrenado en una RTX 4080 local. No se incluye notebook de Colab.

## Resultados

Todas las cifras salen de ejecutar el código del repositorio. Por ahora son **solo sobre fotos sintéticas**: todavía no hay fotos reales anotadas evaluadas.

### Detector (YOLO26n)

De [summary.json](ArtificialVision/results/training/yolo26n_dots_numbers/summary.json): RTX 4080, 29 épocas completadas en 54,5 minutos (el límite de tiempo cortó el entrenamiento) y mejor modelo en la época 27. Métricas en FP32:

| Partición | Clase  | Precisión | Recall | mAP50 | mAP50-95 |
|-----------|--------|-----------|--------|-------|----------|
| val       | todas  | 0,9987    | 0,9986 | 0,995 | 0,9691   |
| val       | dot    | 0,9991    | 0,9992 | 0,995 | 0,9908   |
| val       | number | 0,9983    | 0,9981 | 0,995 | 0,9475   |
| test      | todas  | 0,9992    | 0,9985 | 0,995 | 0,9662   |
| test      | dot    | 0,9993    | 0,9993 | 0,995 | 0,9897   |
| test      | number | 0,9991    | 0,9976 | 0,995 | 0,9428   |

Como referencia, la validación en FP16 da un mAP50-95 de 0,9263 frente a 0,9691 en FP32.

### Pipeline completo, test sintético

200 fotos de `data/synthetic/test/` con 11 261 puntos en total, evaluadas el 2026-11-17 con los comandos de los pasos 4 y 5:

| Modo OCR | F1 puntos | Números leídos | Parejas correctas | Imágenes perfectas | Tiempo medio por foto |
|----------|-----------|----------------|-------------------|--------------------|-----------------------|
| `crops`  | 100 %     | 98,71 %        | 98,71 %           | 82,0 %             | 70,6 ms               |
| `page`   | 100 %     | 92,90 %        | 86,64 %           | 21,5 %             | 701,6 ms              |

El modo `crops` acierta más parejas y es unas 10 veces más rápido. El tiempo se mide desde que el pipeline recibe la foto hasta el emparejamiento, en la RTX 4080 y con los modelos ya cargados; no incluye leer la foto ni guardar los resultados. Estas cifras son anteriores a la incorporación de ArUco, que no cambia nada en las fotos sintéticas porque no llevan la hoja de marcadores.

## Estado y trabajo pendiente

- **Hecho:** generador, formato común, emparejamiento, evaluación, entrenamiento, inferencia con los dos modos de OCR, coordenadas en mm con ArUco y herramientas para sacar fotogramas y etiquetar fotos reales con SAM.
- **Pendiente:**
  - grabar los vídeos de las fichas reales con la cámara del proyecto, etiquetarlos y evaluar,
  - recalibrar la lente con la cámara del proyecto,
  - repartir las fotos reales en entrenamiento, validación y test por vídeo y, según lo que salga, ajustar el generador al encuadre real (16:9, px/mm medidos) y reentrenar,
  - la prueba opcional de robustez con perturbaciones (`common.evaluation` ya admite una carpeta de resultados por perturbación),
  - terminar y evaluar el enfoque de visión clásica en `ClassicVision/` y compararlo con el de visión artificial.

"""Comprueba que el entorno del proyecto está bien montado.

Revisa tres cosas que suelen fallar:
  1. Que PyTorch vea la GPU (si no la ve, se entrenará en CPU, mucho más despacio).
  2. Que Ultralytics y EasyOCR se puedan importar juntos. Las dos librerías instalan OpenCV, en
     variantes distintas, y pueden pisarse.
  3. Que OpenCV pueda abrir ventanas. La variante "headless" no puede, y la herramienta de
     anotación las necesita.

Uso, desde la carpeta del proyecto:
    python tools/check_environment.py         # comprobaciones básicas
    python tools/check_environment.py --ocr   # además lee una imagen de prueba con EasyOCR
                                              # (la primera vez descarga sus modelos)
"""

import argparse
import importlib.metadata as metadata
import sys


def installed_version(package):
    """Devuelve la versión instalada de un paquete o None si no está instalado."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ocr", action="store_true", help="probar también una lectura con EasyOCR")
    args = parser.parse_args()
    errors = []

    # 1. PyTorch y GPU.
    import torch
    print(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        print("  AVISO: PyTorch no ve ninguna GPU; el entrenamiento irá en CPU.")

    # 2. Ultralytics y EasyOCR en el mismo proceso, y qué variantes de OpenCV hay instaladas.
    import cv2
    import easyocr  # noqa: F401  (solo se comprueba que se puede importar)
    import ultralytics
    print(f"Ultralytics {ultralytics.__version__}, EasyOCR {installed_version('easyocr')}, OpenCV {cv2.__version__}")
    variants = {package: installed_version(package)
                for package in ("opencv-python", "opencv-python-headless", "opencv-contrib-python")}
    installed = {package: version for package, version in variants.items() if version}
    print(f"  Variantes de OpenCV instaladas: {installed}")
    if "opencv-python-headless" in installed:
        errors.append("opencv-python-headless está instalado: desinstálalo y reinstala opencv-python.")

    # 3. Ventanas de OpenCV: la variante headless lanza cv2.error al crear una ventana.
    try:
        cv2.namedWindow("check", cv2.WINDOW_NORMAL)
        cv2.waitKey(1)
        cv2.destroyWindow("check")
        print("  OpenCV puede abrir ventanas.")
    except cv2.error as error:
        errors.append(f"OpenCV no puede abrir ventanas: {error}")

    # 4. (Opcional) Lectura real con EasyOCR sobre una imagen sintética con el número 123.
    if args.ocr:
        import numpy as np
        image = np.full((80, 220, 3), 255, dtype=np.uint8)
        cv2.putText(image, "123", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        texts = [text for _, text, _ in reader.readtext(image, allowlist="0123456789")]
        print(f"  EasyOCR ha leído: {texts}")
        if "123" not in texts:
            errors.append("EasyOCR no ha leído '123' en la imagen de prueba.")

    if errors:
        print("\nPROBLEMAS:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print("\nEntorno correcto.")


if __name__ == "__main__":
    main()

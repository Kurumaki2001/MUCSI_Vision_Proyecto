"""Entrenamiento del detector de rotuladores: YOLO26 con una sola clase (0 = rotulador).

Flujo completo, desde la carpeta del proyecto:
    1. Grabar vídeos de los rotuladores sobre la mesa (Documentacion_Rotuladores/video.md) y
       dejarlos en data/pens/videos/
    2. python -m ArtificialVision.pens.frames --every 30            # fotos de los vídeos
    3. python -m ArtificialVision.pens.label_sam                     # etiquetar con SAM (clics)
    4. python -m ArtificialVision.pens.prepare_dataset              # train/val separados por vídeo
    5. python -m ArtificialVision.pens.train                         # este script
    6. python -m ArtificialVision.pens.pipeline foto.jpg             # usar el modelo

Por qué una sola clase y no una por color: YOLO solo tiene que aprender la forma de un rotulador
(necesita menos fotos y no depende del tono de cada tinta) y el color lo decide después el HSV
dentro de la silueta (pens/analysis.py). Si mañana se añade un rotulador verde, basta con añadir
su rango de tono, sin volver a entrenar.

Aumentos de datos: los rotuladores pueden aparecer en cualquier orientación, así que se permiten
volteos horizontal y vertical (a diferencia de los números, aquí no hay texto que leer).

Las métricas y gráficas de Ultralytics quedan en <pens.training.project>/<name>/ (results.png,
matriz de confusión, curvas PR...) y el resumen en summary.json. El mejor modelo se copia a
pens.model.
"""

import json
import shutil
import time
from pathlib import Path

from ArtificialVision.train import metrics_summary
from common.config import load_config, project_path


def main():
    config = load_config()
    cfg, training = config["pens"], config["pens"]["training"]
    data_yaml = project_path(cfg["dataset"]) / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"No existe {data_yaml}. Ejecuta antes python -m ArtificialVision.pens.prepare_dataset")

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Entrenando en la GPU" if device == 0 else "AVISO: sin GPU, se entrena en la CPU (más lento)")
    base_model = project_path(config["training"]["model"])  # el mismo yolo26n.pt preentrenado
    base_model.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(base_model))

    start = time.time()
    model.train(data=str(data_yaml), imgsz=cfg["imgsz"], epochs=training["epochs"], patience=training["patience"],
                batch=training["batch"], seed=training["seed"], deterministic=True,
                fliplr=training["fliplr"], flipud=training["flipud"], degrees=training["degrees"],
                scale=training["scale"], device=device, project=str(project_path(training["project"])),
                name=training["name"], plots=True)
    run_folder = Path(model.trainer.save_dir)
    metrics = model.val(data=str(data_yaml), split="val", imgsz=cfg["imgsz"], device=device,
                        project=str(run_folder), name="val", exist_ok=True, plots=True)
    summary = {"dataset": str(data_yaml), "device": str(device), "completed_epochs": model.trainer.epoch + 1,
               "best_epoch": model.trainer.stopper.best_epoch,
               "training_minutes": round((time.time() - start) / 60, 1), "validation": metrics_summary(metrics)}
    (run_folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    final_model = project_path(cfg["model"])
    shutil.copy2(run_folder / "weights" / "best.pt", final_model)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Modelo copiado a {final_model}. Gráficas en {run_folder}")


# Imprescindible en Windows: los procesos que cargan los datos vuelven a importar este fichero.
if __name__ == "__main__":
    main()

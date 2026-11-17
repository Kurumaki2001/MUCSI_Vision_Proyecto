"""Entrenamiento del detector YOLO26 con dos clases: 0 = dot (punto) y 1 = number (número).

Uso, desde la carpeta del proyecto (antes: python -m data.generate y
python -m ArtificialVision.prepare_yolo):
    python -m ArtificialVision.train           # entrenamiento completo
    python -m ArtificialVision.train --quick   # prueba rápida: dataset mini y 2 épocas

Qué hace:
  1. Parte de yolo26n.pt (preentrenado en COCO) y lo reentrena con las teselas, en la GPU si la
     hay. Con max_hours, Ultralytics recalcula las épocas después de la primera para que el
     entrenamiento completo quepa en ese tiempo.
  2. Ultralytics guarda en <project>/<name>/ las métricas y gráficas (results.csv, results.png,
     matriz de confusión, curvas PR...) y los pesos (weights/best.pt).
  3. Evalúa el mejor modelo en validación y en test, del mismo modo y en precisión completa
     (FP32), y guarda sus gráficas en <project>/<name>/val/ y <project>/<name>/test/.
  4. Escribe summary.json con esas métricas y, en el entrenamiento completo, copia best.pt a
     final_model.

Por qué se vuelve a evaluar al final: durante el entrenamiento, Ultralytics valida en media
precisión (FP16) para ir más rápido. En FP16, una coordenada de entre 512 y 1024 píxeles solo se
puede guardar en pasos de 0,5 píxeles, y con objetos de pocos píxeles eso baja el mAP50-95, que
exige solapes de hasta el 95 %. Por eso las curvas de results.png están algo por debajo de las
cifras de summary.json, que también guarda esa validación en FP16 ("validation_fp16").

Parámetros fijados a propósito (ver config.yaml): imgsz=1024 porque los objetos son pequeños,
fliplr=0 porque los números nunca aparecen en espejo, semilla fija y parada temprana.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from common.config import load_config, project_path


def metrics_summary(metrics):
    """Pasa las métricas de detección de Ultralytics a un diccionario sencillo (global y por clase)."""
    box = metrics.box
    per_class = {}
    for i, class_index in enumerate(box.ap_class_index):
        precision, recall, map50, map50_95 = box.class_result(i)
        per_class[metrics.names[int(class_index)]] = {
            "precision": round(float(precision), 4), "recall": round(float(recall), 4),
            "mAP50": round(float(map50), 4), "mAP50-95": round(float(map50_95), 4)}
    return {"precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4), "mAP50-95": round(float(box.map), 4),
            "per_class": per_class}


def evaluate(model, data_yaml, cfg, device, run_folder):
    """Evalúa el modelo en validación y en test con los mismos ajustes y en precisión completa (FP32).

    Devuelve {"validation": ..., "test": ...} y guarda las gráficas de cada partición en
    run_folder/val y run_folder/test.
    """
    results = {}
    for split, key in (("val", "validation"), ("test", "test")):
        metrics = model.val(data=str(data_yaml), split=split, imgsz=cfg["imgsz"], batch=cfg["batch"],
                            device=device, project=str(run_folder), name=split, exist_ok=True, plots=True)
        results[key] = metrics_summary(metrics)
    return results


def main():
    parser = argparse.ArgumentParser(description="Entrena el detector YOLO26 de puntos y números.")
    parser.add_argument("--quick", action="store_true", help="prueba rápida con el dataset mini y 2 épocas")
    args = parser.parse_args()

    config = load_config()
    cfg = config["training"]
    yolo_folder = project_path(config["quick"]["yolo_output"] if args.quick else config["yolo"]["output"])
    data_yaml = yolo_folder / "data.yaml"
    if not data_yaml.exists():
        option = " --quick" if args.quick else ""
        raise SystemExit(f"No existe {data_yaml}. Ejecuta antes: python -m data.generate{option} "
                         f"y python -m ArtificialVision.prepare_yolo{option}")

    # Se importan aquí para que el aviso anterior salga sin esperar a cargar PyTorch.
    import torch
    from ultralytics import YOLO

    if torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
        print(f"Entrenando en la GPU: {device_name}")
    else:
        device = device_name = "cpu"
        print("AVISO: no hay GPU con CUDA; se entrenará en la CPU, mucho más despacio. "
              "Alternativa: Google Colab con GPU.")

    base_model = project_path(cfg["model"])
    base_model.parent.mkdir(parents=True, exist_ok=True)  # aquí se descarga yolo26n.pt la primera vez
    model = YOLO(str(base_model))

    start = time.time()
    validation_fp16 = model.train(
        data=str(data_yaml),
        imgsz=cfg["imgsz"],
        epochs=config["quick"]["epochs"] if args.quick else cfg["epochs"],
        time=None if args.quick else cfg["max_hours"],
        patience=cfg["patience"],
        batch=cfg["batch"],
        workers=cfg["workers"],
        seed=cfg["seed"],
        deterministic=True,
        fliplr=cfg["fliplr"],
        scale=cfg["scale"],
        degrees=cfg["degrees"],
        device=device,
        project=str(project_path(cfg["project"])),
        name=cfg["name"] + ("_quick" if args.quick else ""),
        exist_ok=args.quick,  # la prueba rápida se sobrescribe; los entrenamientos completos se numeran
        plots=True,
    )
    minutes = (time.time() - start) / 60
    run_folder = Path(model.trainer.save_dir)

    summary = {
        "base_model": cfg["model"],
        "dataset": str(data_yaml),
        "device": device_name,
        "completed_epochs": model.trainer.epoch + 1,
        "best_epoch": model.trainer.stopper.best_epoch,
        "training_minutes": round(minutes, 1),
        # Al acabar train(), 'model' tiene cargado el mejor checkpoint (best.pt).
        **evaluate(model, data_yaml, cfg, device, run_folder),
        "validation_fp16": metrics_summary(validation_fp16),  # la de Ultralytics al acabar de entrenar
    }
    (run_folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.quick:
        final_model = project_path(cfg["final_model"])
        shutil.copy2(run_folder / "weights" / "best.pt", final_model)
        print(f"Modelo final copiado a {final_model}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Resultados y gráficas en {run_folder}")


# Imprescindible en Windows: los procesos que cargan los datos vuelven a importar este fichero y,
# sin esta comprobación, cada uno lanzaría otro entrenamiento.
if __name__ == "__main__":
    main()

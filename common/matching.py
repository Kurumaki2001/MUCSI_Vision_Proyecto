"""Emparejamiento de puntos con números (común a los dos enfoques).

Cada punto se empareja con un número resolviendo un problema de asignación con el algoritmo
húngaro (scipy.optimize.linear_sum_assignment): de todas las formas de formar parejas sin repetir
puntos ni números, se elige la de menor distancia total. Así se resuelven los casos ambiguos en
los que el número más cercano a un punto es en realidad el de su vecino, donde un emparejamiento
voraz (cada punto con su número más cercano) fallaría.

La distancia se mide desde el centro del punto hasta el rectángulo del número, no hasta su
centro: un "118" pegado a su punto tiene el centro lejos, pero el borde muy cerca.
"""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


def point_box_distance(x, y, boxes):
    """Distancia del punto (x, y) al rectángulo [x0, y0, x1, y1]; 0 si el punto cae dentro.

    Admite números sueltos o arrays de NumPy y opera elemento a elemento. Por ejemplo, con un
    punto y un array de cajas de forma (M, 4) devuelve M distancias.
    """
    boxes = np.asarray(boxes, dtype=float)
    dx = np.maximum(np.maximum(boxes[..., 0] - x, 0.0), x - boxes[..., 2])
    dy = np.maximum(np.maximum(boxes[..., 1] - y, 0.0), y - boxes[..., 3])
    return np.hypot(dx, dy)


def assign(cost, max_cost):
    """Algoritmo húngaro con un coste máximo admitido.

    Devuelve las parejas (fila, columna) de la asignación de coste mínimo, quitando las que
    superan max_cost. Las parejas no admitidas no se marcan con infinito (eso puede dejar el
    problema sin solución), sino con un coste mayor que la suma de cualquier asignación admitida.
    Así el algoritmo primero forma el máximo número de parejas admitidas y, entre esas
    opciones, minimiza el coste total. Al final se descartan las no admitidas que haya usado.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    forbidden = cost > max_cost
    forbidden_cost = max_cost * (min(cost.shape) + 1) + 1
    rows, cols = linear_sum_assignment(np.where(forbidden, forbidden_cost, cost))
    return [(int(i), int(j)) for i, j in zip(rows, cols) if not forbidden[i, j]]


def max_pair_distance(numbers, factor):
    """Distancia máxima para que un punto y un número formen pareja: factor x altura mediana de los números.

    Se mide en alturas de número y no en píxeles, para que valga con fotos de cualquier
    resolución. Devuelve None si no hay números con caja.
    """
    heights = [number["box"][3] - number["box"][1] for number in numbers if number.get("box")]
    return factor * float(np.median(heights)) if heights else None


def match_dots_numbers(dots, numbers, max_distance):
    """Empareja puntos y números con el algoritmo húngaro y la distancia punto-rectángulo.

    dots: lista de {"id", "x", "y", ...}; numbers: lista de {"id", "box", "value", ...}. Un número
    sin valor (value None, porque no se ha podido leer) participa igual: si se quitara, su punto
    podría quedarse con el número de un vecino. Por encima de max_distance no hay pareja.

    Devuelve (pairs, unpaired_dots, unpaired_numbers): pairs es una lista de
    {"dot", "number", "distance"} y las otras dos son listas de ids.
    """
    if not dots or not numbers:
        return [], [dot["id"] for dot in dots], [number["id"] for number in numbers]
    xs = np.array([dot["x"] for dot in dots], dtype=float)
    ys = np.array([dot["y"] for dot in dots], dtype=float)
    boxes = np.array([number["box"] for number in numbers], dtype=float)
    # distances[i, j] = distancia del punto i a la caja del número j
    distances = point_box_distance(xs[:, None], ys[:, None], boxes[None, :, :])

    pairs = [{"dot": dots[i]["id"], "number": numbers[j]["id"], "distance": round(float(distances[i, j]), 2)}
             for i, j in assign(distances, max_distance)]
    paired_dots = {pair["dot"] for pair in pairs}
    paired_numbers = {pair["number"] for pair in pairs}
    unpaired_dots = [dot["id"] for dot in dots if dot["id"] not in paired_dots]
    unpaired_numbers = [number["id"] for number in numbers if number["id"] not in paired_numbers]
    return pairs, unpaired_dots, unpaired_numbers


def sequence_check(values, min_coverage=0.75):
    """Diagnóstico: si los números forman una secuencia 1..N, cuáles faltan, cuáles se repiten y
    cuáles se salen de ella.

    N es el mayor valor leído tal que de 1 a N aparece al menos la fracción min_coverage de los
    números. Así, un número mal leído (un "140" en una ficha del 1 al 98) no estira la secuencia:
    queda como fuera de rango, que suele indicar una lectura errónea. Si ningún valor cumple la
    condición (por ejemplo, si solo se ve un trozo de la ficha, del 87 al 120), devuelve None.
    Es solo informativo: no cambia el emparejamiento.
    """
    known = [value for value in values if value is not None and value >= 1]
    present = set(known)
    # En orden ascendente, la posición indica cuántos valores distintos se han leído hasta v.
    n = max((v for count, v in enumerate(sorted(present), start=1) if count / v >= min_coverage),
            default=None)
    if n is None:
        return None
    counts = Counter(value for value in known if value <= n)
    return {"n": n,
            "missing": [value for value in range(1, n + 1) if value not in present],
            "repeated": sorted(value for value, count in counts.items() if count > 1),
            "out_of_range": sorted(value for value in present if value > n)}

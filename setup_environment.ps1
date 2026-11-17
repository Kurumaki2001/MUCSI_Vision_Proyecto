# Crea el entorno virtual .venv del proyecto e instala las dependencias.
#
# Uso, desde la carpeta del proyecto:
#   powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1         # con GPU NVIDIA (CUDA)
#   powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1 -NoGpu  # solo CPU
param([switch]$NoGpu)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venvPath = Join-Path $root ".venv"
$python = Join-Path $venvPath "Scripts\python.exe"
$requirements = Join-Path $root "requirements.txt"

# 1. Entorno virtual con Python 3.12 (con el lanzador py si existe; si no, con el python del PATH).
if (-not (Test-Path $python)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & py -3.12 -m venv $venvPath
    }
    else {
        $version = & python --version
        if ($version -notmatch "^Python 3\.12\.") {
            throw "Se requiere Python 3.12 (encontrado: $version)."
        }
        & python -m venv $venvPath
    }
    if ($LASTEXITCODE -ne 0) { throw "No se ha podido crear el entorno virtual." }
}
& $python -m pip install --upgrade pip

# 2. PyTorch con CUDA (salvo con -NoGpu). El índice cu132 sirve para drivers con CUDA 13.2 o superior.
if (-not $NoGpu) {
    & $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
    if ($LASTEXITCODE -ne 0) { throw "No se ha podido instalar PyTorch con CUDA." }
}

# 3. Resto de dependencias.
& $python -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "No se han podido instalar las dependencias de requirements.txt." }

# 4. OpenCV. EasyOCR instala opencv-python-headless (sin ventanas) y Ultralytics, opencv-python.
#    Las dos versiones escriben el mismo módulo cv2 y la última que se instala pisa a la otra,
#    así que se quitan ambas y se reinstala solo opencv-python, que sí puede abrir ventanas.
$openCvLine = (Select-String -Path $requirements -Pattern "^opencv-python==").Line
& $python -m pip uninstall -y opencv-python-headless opencv-python
& $python -m pip install --no-deps $openCvLine
if ($LASTEXITCODE -ne 0) { throw "No se ha podido reinstalar $openCvLine." }

# 5. Comprobación final (GPU, convivencia de EasyOCR y Ultralytics, ventanas de OpenCV).
& $python (Join-Path $root "tools\check_environment.py")
if ($LASTEXITCODE -ne 0) { throw "La comprobación del entorno ha fallado." }

Write-Host "Entorno preparado en $venvPath"

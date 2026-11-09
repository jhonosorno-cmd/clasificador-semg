@echo off
setlocal EnableExtensions
REM Prepara el entorno para el experimento LOSO (Windows).
REM Uso: experimento_loso\setup.bat
REM Variables opcionales:
REM   TORCH_INDEX_URL  indice de PyTorch para CUDA (por defecto cu126)
REM   SKIP_SMOKE=1     omite la prueba rapida del experimento

REM Raiz del repositorio = carpeta padre de la carpeta de este script.
cd /d "%~dp0.."
if %ERRORLEVEL% neq 0 (
    echo [ERROR] No se pudo acceder a la raiz del repositorio.
    exit /b 1
)

echo ============================================================
echo  Setup del experimento LOSO - Clasificador SEMG ^(NinaPro DB3^)
echo  Raiz: %CD%
echo ============================================================

REM 2. Buscar Python 3.10 o superior: primero py -3, luego python.
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if %ERRORLEVEL% equ 0 set "PY=py -3"
if defined PY goto :python_ok
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if %ERRORLEVEL% equ 0 set "PY=python"
if not defined PY (
    echo [ERROR] No se encontro Python 3.10 o superior ^(probados: py -3 y python^).
    echo         Instala Python 3.10+ desde python.org y reintenta.
    exit /b 1
)
:python_ok
echo [INFO] Python usado: %PY%

REM 3. Crear el entorno virtual .venv en la raiz y actualizar pip.
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creando entorno virtual .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual .venv.
        exit /b 1
    )
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip
if %ERRORLEVEL% neq 0 (
    echo [ERROR] No se pudo actualizar pip.
    exit /b 1
)

REM 4. Instalar PyTorch: CUDA si hay nvidia-smi, CPU en otro caso.
"%VPY%" -c "import torch" >nul 2>&1
if %ERRORLEVEL% equ 0 goto :torch_ya
set "TORCH_CUDA_URL=https://download.pytorch.org/whl/cu126"
if defined TORCH_INDEX_URL set "TORCH_CUDA_URL=%TORCH_INDEX_URL%"
set "TORCH_URL=https://download.pytorch.org/whl/cpu"
set "TORCH_VARIANTE=CPU"
nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 set "TORCH_URL=%TORCH_CUDA_URL%"
if %ERRORLEVEL% equ 0 set "TORCH_VARIANTE=CUDA"
echo [INFO] Variante elegida: %TORCH_VARIANTE% ^(indice: %TORCH_URL%^)
"%VPY%" -m pip install torch --index-url %TORCH_URL%
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Fallo la instalacion de PyTorch.
    exit /b 1
)
goto :torch_fin
:torch_ya
"%VPY%" -c "import torch; print('[INFO] PyTorch ya instalado:', torch.__version__, '- cuda.is_available =', torch.cuda.is_available())"
:torch_fin

REM 5. Dependencias del proyecto y pytest.
"%VPY%" -m pip install -r requirements.txt pytest
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Fallo la instalacion de requirements.txt.
    exit /b 1
)

REM 6. Descargar y verificar los 11 archivos .mat de NinaPro DB3.
if not exist "experimento_loso\download_data.py" (
    echo [ERROR] Falta experimento_loso\download_data.py.
    exit /b 1
)
echo [INFO] Descargando y verificando datos ^(data\raw^) ...
"%VPY%" experimento_loso\download_data.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Fallo la descarga o verificacion de los datos.
    exit /b 1
)

REM 7. Pruebas unitarias: abortan el setup si fallan.
echo [INFO] Ejecutando pruebas ...
"%VPY%" -m pytest tests\ -q
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Fallaron las pruebas.
    exit /b 1
)

REM 8. Prueba rapida del experimento (se puede omitir con set SKIP_SMOKE=1).
if "%SKIP_SMOKE%"=="1" (
    echo [INFO] Prueba rapida omitida ^(SKIP_SMOKE=1^).
    goto :final
)
echo [INFO] Prueba rapida del experimento ...
for /f "delims=" %%t in ('powershell -NoProfile -Command "Get-Date -Format o"') do set "T_INI=%%t"
"%VPY%" experimento_loso\run_loso.py --models v10 --subjects 1 --conditions zero --epochs 1 --ft-epochs 1 --max-batches 5 --force
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Fallo la prueba rapida del experimento.
    exit /b 1
)
for /f "delims=" %%e in ('powershell -NoProfile -Command "[int]((Get-Date) - [datetime]::Parse($env:T_INI)).TotalSeconds"') do set "SEGUNDOS=%%e"
echo [INFO] Prueba rapida completada en %SEGUNDOS% s.

REM Eliminar las salidas de la prueba.
if exist "experimento_loso\resultados\v10_S1_zero_seed42.json" del /q "experimento_loso\resultados\v10_S1_zero_seed42.json"
if exist "experimento_loso\resultados\probs\v10_S1_zero_seed42.npz" del /q "experimento_loso\resultados\probs\v10_S1_zero_seed42.npz"
if exist "experimento_loso\checkpoints\v10_S1_pretrain_seed42.pth" del /q "experimento_loso\checkpoints\v10_S1_pretrain_seed42.pth"

REM 9. Mensaje final.
:final
echo.
echo ============================================================
echo  Listo: entorno preparado para el experimento LOSO
echo.
echo  1. Activa el entorno virtual:
echo       .venv\Scripts\activate
echo  2. Lanza el experimento ^(con el venv activo^), por ejemplo:
echo       experimento_loso\run_all.bat 1 2 3 4
echo.
echo  run_all debe ejecutarse con el venv activo.
echo ============================================================
exit /b 0

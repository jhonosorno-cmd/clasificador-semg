@echo off
setlocal

rem Ejecuta el experimento leave-one-subject-out por sujeto (reanudable).
rem
rem Uso (desde cualquier directorio):
rem   experimento_loso\run_all.bat 1 2 3 4
rem   set MODELS=v10 v17 && experimento_loso\run_all.bat 5 6
rem   set EXTRA_ARGS=--seed 7 && experimento_loso\run_all.bat 1
rem
rem Sin argumentos ejecuta los sujetos 1 a 11.

cd /d "%~dp0.."

if "%~1"=="" (
    set "SUBJECTS=1 2 3 4 5 6 7 8 9 10 11"
) else (
    set "SUBJECTS=%*"
)
if not defined MODELS set "MODELS=v10 v15 v17 v18"
if not defined EXTRA_ARGS set "EXTRA_ARGS="

echo Inicio: %DATE% %TIME%
echo Directorio: %CD%
echo Sujetos: %SUBJECTS%
echo Modelos: %MODELS%

for %%S in (%SUBJECTS%) do (
    echo --- Sujeto %%S ---
    python experimento_loso\run_loso.py --models %MODELS% --subjects %%S --conditions zero ft1 ft2 --seed 42 %EXTRA_ARGS%
    if errorlevel 1 (
        echo Error en el sujeto %%S. Vuelva a lanzar el mismo comando para reanudar.
        exit /b 1
    )
)

echo Fin: %DATE% %TIME%
endlocal

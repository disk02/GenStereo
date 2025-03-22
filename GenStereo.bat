batch
Copy
@echo off
setlocal enabledelayedexpansion

:: GenStereo Batch Processor with Conda Activation
:: ----------------------------------------------

echo.
echo  #################################################
echo  ##          GenStereo Batch Processor          ##
echo  #################################################
echo.

:: Activate Conda environment
call conda activate genstereo
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to activate conda environment 'genstereo'
    echo 1. Ensure conda is installed
    echo 2. Create environment with: conda create -n genstereo python=3.10
    echo 3. Install requirements: pip install -r requirements.txt
    pause
    exit /b 1
)

:: Set text colors
set RED=1
set GREEN=2
set YELLOW=3
set BLUE=4
set MAGENTA=5
set CYAN=6

:: Prompt for parameters
:input_path
echo.
echo [Input Configuration]
set /p "input_path=Enter input file/directory path: "
if not exist "%input_path%" (
    echo.
    echo ERROR: Input path does not exist!
    goto input_path
)

:output_dir
echo.
set /p "output_dir=Enter output directory: "
if "%output_dir%"=="" (
    echo ERROR: Output directory required!
    goto output_dir
)

:: Create output directory if needed
if not exist "%output_dir%" (
    echo Creating output directory...
    mkdir "%output_dir%"
)

:scale_factor
echo.
set /p "scale=Enter scale factor [15.0]: "
if "%scale%"=="" set scale=15.0

:tile_size
echo.
set /p "tile_size=Enter tile size [512]: "
if "%tile_size%"=="" set tile_size=512

:: Confirmation
echo.
echo #################################################
echo # Selected Configuration:
echo # Input:    %input_path%
echo # Output:   %output_dir%
echo # Scale:    %scale%
echo # Tile Size: %tile_size%
echo #################################################
echo.

choice /c YN /n /m "Start processing with these settings? (Y/N): "
if errorlevel 2 goto cancel

:: Run processing
echo.
echo Starting processing...
echo -------------------------------------------------

python genstereo_cli.py "%input_path%" "%output_dir%" --scale %scale% --tile_size %tile_size%

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Processing failed! Check paths and parameters
    pause
    exit /b 1
)

echo.
echo Processing completed successfully!
echo Results saved to: %output_dir%
pause
exit /b 0

:cancel
echo.
echo Processing cancelled by user
pause
exit /b 0
@echo off
cd /d "%~dp0"
echo Installing dependencies...
python -m pip install -r requirements.txt
echo.
echo 1/6 Data quality check
python 00_data_quality_check.py
echo.
echo 2/6 Data inspection
python 01_data_inspection.py
echo.
echo 3/6 Preprocessing
python 02_data_preprocessing.py
echo.
echo 4/6 Trend and polynomial analysis
python 03_trend_analysis.py
python 04_polynomial_regression.py
echo.
echo 5/6 Clustering and NFHS analysis
python 05_clustering.py
python 06_cluster_analysis.py
python 06_nfhs_analysis.py
echo.
echo 6/6 Pooled chronological regression
python 07_model_comparison.py
echo.
echo Final summary
python 08_final_results.py
pause

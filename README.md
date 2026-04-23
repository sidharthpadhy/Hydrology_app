# Online Hydrology Application (Drone Data)

This starter app provides a web UI to upload terrain and hydrology inputs and generate:

- Watershed raster
- Stream raster
- Longest stream raster
- Flood depth rasters for multiple return periods

## Inputs supported

- **Drone DEM** (GeoTIFF, required)
- **Rainfall** CSV with columns: `return_period,rainfall_mm` (optional)
- **LULC raster** (GeoTIFF, optional)
- **Soil raster** (GeoTIFF, optional)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Rainfall CSV example

```csv
return_period,rainfall_mm
10,70
25,100
50,130
100,170
```

## Notes

- The hydrology engine in `hydrology_app/core.py` is a lightweight, transparent baseline (D8-like terrain routing).
- For production-scale studies, connect this UI to calibrated models (e.g., HEC-HMS/HEC-RAS, LISFLOOD-FP, SWAT+, etc.) and validated geospatial workflows.

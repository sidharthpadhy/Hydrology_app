from __future__ import annotations

import io
from typing import Dict, Tuple

import numpy as np
import streamlit as st

from hydrology_app.core import run_hydrology

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None


st.set_page_config(page_title="Hydrology App", layout="wide")
st.title("Drone-driven Hydrology Analysis")
st.caption(
    "Upload drone DEM and optional rainfall/LULC/soil datasets to generate watershed, stream network, "
    "longest stream and flood-depth rasters for return periods."
)


def read_raster(uploaded_file) -> Tuple[np.ndarray, dict | None]:
    if uploaded_file is None:
        return None, None
    if rasterio is None:
        st.error("rasterio is not installed. Please install dependencies from requirements.txt")
        st.stop()
    with rasterio.open(uploaded_file) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
    return arr, profile


def parse_rainfall(file) -> Dict[int, float]:
    if file is None:
        return {10: 70.0, 25: 100.0, 50: 130.0, 100: 170.0}

    content = file.read().decode("utf-8").strip().splitlines()
    out: Dict[int, float] = {}
    for line in content[1:]:
        rp, rainfall = line.split(",")
        out[int(rp.strip())] = float(rainfall.strip())
    return out


def write_geotiff(array: np.ndarray, profile: dict) -> bytes:
    if rasterio is None:
        return b""
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, compress="lzw")
    mem = io.BytesIO()
    with rasterio.io.MemoryFile() as mfile:
        with mfile.open(**out_profile) as dst:
            dst.write(array.astype(np.float32), 1)
        mem.write(mfile.read())
    return mem.getvalue()


with st.sidebar:
    st.header("Input data")
    dem_file = st.file_uploader("Drone DEM (GeoTIFF)", type=["tif", "tiff"])
    rainfall_file = st.file_uploader(
        "Rainfall CSV (return_period,rainfall_mm)",
        type=["csv"],
        help="CSV header must be: return_period,rainfall_mm",
    )
    lulc_file = st.file_uploader("LULC raster (optional, GeoTIFF)", type=["tif", "tiff"])
    soil_file = st.file_uploader("Soil raster (optional, GeoTIFF)", type=["tif", "tiff"])
    run_button = st.button("Run hydrology workflow", type="primary")

if run_button:
    if dem_file is None:
        st.warning("Please upload a drone DEM file.")
        st.stop()

    dem, dem_profile = read_raster(dem_file)
    lulc, _ = read_raster(lulc_file) if lulc_file else (None, None)
    soil, _ = read_raster(soil_file) if soil_file else (None, None)
    rain_depths = parse_rainfall(rainfall_file)

    with st.spinner("Running terrain and flood analysis..."):
        outputs = run_hydrology(dem, rain_depths, lulc=lulc, soil=soil)

    st.success("Hydrology analysis complete")

    tab1, tab2, tab3 = st.tabs(["Terrain outputs", "Flood outputs", "Downloads"])

    with tab1:
        c1, c2 = st.columns(2)
        c1.subheader("Watershed")
        c1.image(outputs.watershed_mask.astype(np.uint8) * 255, clamp=True)
        c2.subheader("Streams and longest stream")
        stream_rgb = np.zeros((*outputs.stream_mask.shape, 3), dtype=np.uint8)
        stream_rgb[outputs.stream_mask] = (0, 140, 255)
        stream_rgb[outputs.longest_stream_mask] = (255, 70, 70)
        c2.image(stream_rgb)

        st.subheader("Flow accumulation")
        st.image(np.log1p(outputs.flow_accumulation), clamp=True)

    with tab2:
        rp = st.selectbox("Return period", sorted(outputs.flood_depths))
        flood = outputs.flood_depths[rp]
        st.metric("Max flood depth (m)", f"{float(flood.max()):.3f}")
        st.image(flood, clamp=True)

    with tab3:
        watershed_tif = write_geotiff(outputs.watershed_mask.astype(np.float32), dem_profile)
        streams_tif = write_geotiff(outputs.stream_mask.astype(np.float32), dem_profile)
        longest_tif = write_geotiff(outputs.longest_stream_mask.astype(np.float32), dem_profile)

        st.download_button("Download watershed raster", watershed_tif, "watershed.tif")
        st.download_button("Download streams raster", streams_tif, "streams.tif")
        st.download_button("Download longest stream raster", longest_tif, "longest_stream.tif")

        for rp, arr in outputs.flood_depths.items():
            flood_tif = write_geotiff(arr, dem_profile)
            st.download_button(
                f"Download flood depth {rp}-year raster",
                flood_tif,
                f"flood_depth_{rp}yr.tif",
                key=f"download-flood-{rp}",
            )
else:
    st.info("Upload inputs from the sidebar and click **Run hydrology workflow**.")

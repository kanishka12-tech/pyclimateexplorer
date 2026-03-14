# pyclimateexplorer
Interactive climate data visualization dashboard using Python and NetCDF datasets
# PyClimaExplorer 🌍

PyClimaExplorer is an interactive climate data visualization dashboard built using Python and Streamlit. It allows users to explore NetCDF climate datasets through spatial and temporal visualizations.

## Features

- Global climate spatial visualization
- Time-series climate analysis
- Interactive maps using Plotly
- Support for NetCDF climate datasets
- Fast data loading with Xarray

## Tech Stack

Python  
Streamlit  
Xarray  
NumPy  
Pandas  
Plotly

## Project Structure

```
modules/
    data_loader.py
    spatial_view.py
    temporal_view.py

data/
    sample_climate_data.nc

app.py
```

## Run the Project

Install dependencies

```
pip install -r requirements.txt
```

Run the dashboard

```
streamlit run app.py
```

## Dataset

Sample climate data is provided in `.nc` NetCDF format.

## Use Case

Helps researchers quickly visualize climate variables like:

- Temperature
- Precipitation
- Wind speed

## Future Improvements

- Multi-dataset support
- Climate anomaly detection
- AI based climate pattern insights

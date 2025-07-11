import streamlit as st
import numpy as np
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from clathrate_analysis import (
    parse_particles_and_shape, plot_clathrate_cavities, analyze_clathrate_structure,
    get_bipyramid_geometry, get_truncated_bipyramid_geometry
)
from truncation_analysis import plot_truncation_cavity_comparison

st.set_page_config(page_title="Clathrate Cavity Explorer", layout="wide")
st.title("Clathrate/Truncated Bipyramid Cavity Explorer")

# --- File selection/upload ---
st.sidebar.header("1. Structure File")
default_file = "models/Cages/Cage A new.pos"
file = st.sidebar.file_uploader("Upload .pos file", type=["pos"])

if file is not None:
    # Save uploaded file to a temp location
    temp_path = "uploaded_structure.pos"
    with open(temp_path, "wb") as f:
        f.write(file.read())
    filename = temp_path
else:
    filename = default_file
    st.sidebar.info(f"Using default: {default_file}")

# --- Truncation factor ---
st.sidebar.header("2. Truncation Factor")
truncation = st.sidebar.slider("Truncation (t)", 0.0, 0.8, 0.0, 0.05)

# --- Voxel grid size ---
st.sidebar.header("3. Voxel Grid Size")
grid_size = st.sidebar.slider("Grid Size", 32, 128, 64, 8)

# --- Duplication controls ---
st.sidebar.header("4. Unit Cell Duplication")
nx = st.sidebar.number_input("Number of cells in X", min_value=1, max_value=5, value=1, step=1)
ny = st.sidebar.number_input("Number of cells in Y", min_value=1, max_value=5, value=1, step=1)
nz = st.sidebar.number_input("Number of cells in Z", min_value=1, max_value=5, value=1, step=1)

# --- Run button ---
run = st.sidebar.button("Run Analysis")

# --- Main logic ---
if run or (file is None):
    st.write(f"**File:** `{filename}`")
    st.write(f"**Truncation:** {truncation}")
    st.write(f"**Grid Size:** {grid_size}")
    
    # Parse file
    try:
        particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(filename)
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        st.stop()
    
    # Choose geometry
    if truncation > 0:
        geometry_func = get_truncated_bipyramid_geometry
        geom_label = "Truncated Bipyramid"
    else:
        geometry_func = get_bipyramid_geometry
        geom_label = "Regular Bipyramid"
    
    # Duplicate unit cell if needed
    from show_model_clean import duplicate_unit_cell
    if nx > 1 or ny > 1 or nz > 1:
        duplicated_particles = duplicate_unit_cell(particles, nx, ny, nz, simulation_data)
        # Remove cell indices for downstream functions
        all_particles = [(pos, quat) for pos, quat, _ in duplicated_particles]
        duplication_label = f"{nx} x {ny} x {nz}"
    else:
        all_particles = particles
        duplication_label = "1 x 1 x 1"

    # Detect cavities
    with st.spinner("Detecting cavities and analyzing structure..."):
        cavity_centers, cavity_volumes, cavity_radii = plot_clathrate_cavities(
            particles=all_particles,
            shape_vertices=shape_vertices,
            shape_color=shape_color,
            geometry_func=geometry_func,
            truncation_factor=truncation if truncation > 0 else None,
            show_particles=True,
            show_cavities=True,
            show_spheres=True,
            simulation_data=simulation_data,
        )
        analysis = analyze_clathrate_structure(
            particles=all_particles,
            shape_vertices=shape_vertices,
            shape_color=shape_color,
            geometry_func=geometry_func,
            truncation_factor=truncation if truncation > 0 else None,
            simulation_data=simulation_data
        )
    
    # --- Display stats ---
    st.subheader("Cavity Analysis Results")
    st.write(f"**Geometry:** {geom_label}")
    st.write(f"**Unit cell duplication:** {duplication_label}")
    st.write(f"**Number of cavities:** {len(cavity_centers)}")
    st.write(f"**Total cavity volume:** {np.sum(cavity_volumes):.6f}")
    st.write(f"**Average cavity radius:** {np.mean(cavity_radii) if cavity_radii else 0:.6f}")
    st.write(f"**Cavity fraction:** {analysis['cavity_fraction']*100:.2f}%")
    
    # --- Show 3D plot (Plotly) ---
    st.subheader("3D Visualization")
    st.info("Rotate/zoom the plot below. Spheres show largest inscribed cavity at each center.")
    # plot_clathrate_cavities already shows the plot, but we want to embed it in Streamlit
    # So, we need to return the Plotly figure instead (modify plot_clathrate_cavities if needed)
    st.warning("3D visualization will appear in a separate window if not embedded. For full integration, refactor plot_clathrate_cavities to return a Plotly figure.")
    
    # Optionally: plot_truncation_cavity_comparison for a range of t
    if st.checkbox("Show cavity evolution with truncation (comparison plot)"):
        st.write("This may take a few seconds...")
        plot_truncation_cavity_comparison(
            particles=particles,
            shape_vertices=shape_vertices,
            shape_color=shape_color,
            truncation_factors=[0.0, 0.2, 0.4, 0.6, 0.8],
            simulation_data=simulation_data
        )

st.sidebar.markdown("---")
st.sidebar.info("Developed with Streamlit. For advanced features, edit clathrate_gui.py.") 
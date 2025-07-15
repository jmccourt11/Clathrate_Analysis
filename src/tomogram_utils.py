"""
Tomogram and TIFF Utilities Module

This module provides functions for creating, saving, loading, and visualizing
3D tomograms from particle assemblies.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from clathrate_analysis import voxelize_particles
from scipy.spatial import ConvexHull
from clathrate_analysis import plot_cavity_objects

def save_voxel_grid_as_tiff(voxel_grid, filename, pixel_size=1.0):
    """
    Save 3D voxel grid as a multi-page TIFF file (tomogram format).
    
    Args:
        voxel_grid: 3D numpy array (z, y, x) format
        filename: output filename (should end with .tif or .tiff)
        pixel_size: physical size of each pixel in nanometers
    """
    try:
        import tifffile
    except ImportError:
        print("tifffile not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "tifffile"])
        import tifffile
    
    # Ensure voxel_grid is in the correct format (z, y, x)
    if len(voxel_grid.shape) != 3:
        raise ValueError("voxel_grid must be a 3D array")
    
    # Convert to appropriate data type for TIFF
    if voxel_grid.dtype != np.float32:
        voxel_grid = voxel_grid.astype(np.float32)
    
    # Save as multi-page TIFF
    tifffile.imwrite(
        filename,
        voxel_grid,
        imagej=True,  # Use ImageJ format for better compatibility
        photometric='minisblack',
        planarconfig='contig'
    )
    
    print(f"Saved 3D tomogram as: {filename}")
    print(f"Dimensions: {voxel_grid.shape[2]}x{voxel_grid.shape[1]}x{voxel_grid.shape[0]} pixels")
    print(f"Pixel size: {pixel_size} nm")
    print(f"File size: {voxel_grid.nbytes / (1024**2):.1f} MB")

def load_tomogram_tiff(filename):
    """
    Load a 3D tomogram from TIFF file.
    
    Args:
        filename: path to TIFF file
    
    Returns:
        voxel_grid: 3D numpy array
    """
    try:
        import tifffile
    except ImportError:
        print("tifffile not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "tifffile"])
        import tifffile
    
    # Load the TIFF file
    voxel_grid = tifffile.imread(filename)
    
    print(f"Loaded tomogram: {filename}")
    print(f"Dimensions: {voxel_grid.shape}")
    print(f"Data type: {voxel_grid.dtype}")
    print(f"Value range: {voxel_grid.min():.3f} to {voxel_grid.max():.3f}")
    
    return voxel_grid

def create_tomogram_from_particles(particles, grid_size=128, padding=0.1, shape_vertices=None, shape_faces=None, 
                                 pixel_size=1.0, filename=None):
    """
    Create and save a 3D tomogram from particle positions.
    
    Args:
        particles: list of (position, quaternion) tuples
        grid_size: number of voxels per dimension
        padding: fraction of box size to pad on each side
        shape_vertices: (N, 3) array of particle vertices
        shape_faces: list of face indices
        pixel_size: physical size of each pixel in nanometers
        filename: output filename (if None, auto-generate)
    
    Returns:
        filename: path to saved file
    """

    # Voxelize particles
    print("Voxelizing particles...")
    voxel_grid, edges = voxelize_particles(particles, grid_size, padding, shape_vertices, shape_faces)
    
    # Auto-generate filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tomogram_{timestamp}.tif"
    
    # Save as TIFF
    save_voxel_grid_as_tiff(voxel_grid, filename, pixel_size)
    
    return filename

def create_tomogram_with_cavity_objects(particles, cavity_centers, cavity_radii, 
                                  grid_size=128, padding=0.1, shape_vertices=None, shape_faces=None,
                                  pixel_size=1.0, filename=None, geometry_func=None, truncation_factor=None,
                                  cavity_object_type='cube', cavity_object_scale=1.0):
    """
    Create and save a 3D tomogram from particle positions with objects placed in cavities.
    
    Args:
        particles: list of (position, quaternion) tuples
        cavity_centers: List of cavity center coordinates
        cavity_radii: List of cavity radii
        grid_size: number of voxels per dimension
        padding: fraction of box size to pad on each side
        shape_vertices: (N, 3) array of particle vertices
        shape_faces: list of face indices
        pixel_size: physical size of each pixel in nanometers
        filename: output filename (if None, auto-generate)
        geometry_func: function to get geometry for particles
        truncation_factor: truncation parameter for particles
        cavity_object_type: Type of object to place at cavities ('cube', 'bipyramid', etc.)
        cavity_object_scale: Scale factor to adjust object size relative to cavity radius
    
    Returns:
        filename: path to saved file
    """
    # Get voxel grid from plot_cavity_objects
    print(f"Creating voxel grid with particles and cavity {cavity_object_type}s...")
    voxel_grid, _ = plot_cavity_objects(
        particles=particles,
        cavity_centers=cavity_centers,
        cavity_radii=cavity_radii,
        shape_vertices=shape_vertices,
        show_particles=True,
        geometry_func=geometry_func,
        truncation_factor=truncation_factor,
        cavity_object_type=cavity_object_type,
        cavity_object_scale=cavity_object_scale
    )
    
    # Auto-generate filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tomogram_with_{cavity_object_type}s_{timestamp}.tif"
    
    # Save as TIFF
    save_voxel_grid_as_tiff(voxel_grid, filename, pixel_size)
    
    print(f"\nTomogram statistics:")
    print(f"  Grid size: {grid_size}x{grid_size}x{grid_size}")
    print(f"  Original particles: {np.sum(voxel_grid == 1.0)} voxels")
    print(f"  Cavity {cavity_object_type}s: {np.sum(voxel_grid == 2.0)} voxels")
    print(f"  Empty space: {np.sum(voxel_grid == 0.0)} voxels")
    
    return filename

def plot_3d_tomogram(filename, plot_type='isosurface', threshold=0.5, opacity=0.7, colorscale='Viridis'):
    """
    Load a tomogram and create a 3D plot.
    
    Args:
        filename: path to TIFF tomogram file
        plot_type: 'isosurface', 'volume', or 'slices'
        threshold: threshold value for isosurface (0-1)
        opacity: opacity for volume rendering (0-1)
        colorscale: colorscale for the plot
    """
    # Load the tomogram
    print(f"Loading tomogram: {filename}")
    voxel_grid = load_tomogram_tiff(filename)
    
    # Normalize the data to 0-1 range
    voxel_grid_norm = (voxel_grid - voxel_grid.min()) / (voxel_grid.max() - voxel_grid.min())
    
    if plot_type == 'isosurface':
        plot_3d_isosurface(voxel_grid_norm, threshold, colorscale)
    elif plot_type == 'volume':
        plot_3d_volume(voxel_grid_norm, opacity, colorscale)
    elif plot_type == 'slices':
        plot_3d_slices(voxel_grid_norm, colorscale)
    else:
        raise ValueError("plot_type must be 'isosurface', 'volume', or 'slices'")

def plot_3d_isosurface(voxel_grid, threshold=0.5, colorscale='Viridis'):
    """
    Create a 3D isosurface plot of the tomogram.
    
    Args:
        voxel_grid: 3D numpy array (normalized to 0-1)
        threshold: threshold value for isosurface (0-1)
        colorscale: colorscale for the plot
    """
    # Create coordinate grids
    z, y, x = np.mgrid[0:voxel_grid.shape[0], 0:voxel_grid.shape[1], 0:voxel_grid.shape[2]]
    
    # Create isosurface
    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=voxel_grid.flatten(),
        isomin=threshold,
        isomax=1.0,
        opacity=0.7,
        surface_count=1,
        colorscale=colorscale,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    fig.update_layout(
        title=f'3D Tomogram Isosurface (threshold={threshold})',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False
    )
    
    fig.show()

def plot_3d_volume(voxel_grid, opacity=0.7, colorscale='Viridis'):
    """
    Create a 3D volume rendering of the tomogram.
    
    Args:
        voxel_grid: 3D numpy array (normalized to 0-1)
        opacity: opacity for volume rendering (0-1)
        colorscale: colorscale for the plot
    """
    # Create coordinate grids
    z, y, x = np.mgrid[0:voxel_grid.shape[0], 0:voxel_grid.shape[1], 0:voxel_grid.shape[2]]
    
    # Create volume plot
    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=voxel_grid.flatten(),
        opacity=opacity,
        colorscale=colorscale,
        surface_count=15,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    fig.update_layout(
        title='3D Tomogram Volume Rendering',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False
    )
    
    fig.show()

def plot_3d_slices(voxel_grid, colorscale='Viridis'):
    """
    Create a 3D plot showing orthogonal slices of the tomogram.
    
    Args:
        voxel_grid: 3D numpy array (normalized to 0-1)
        colorscale: colorscale for the plot
    """
    # Get center slices
    center_z = voxel_grid.shape[0] // 2
    center_y = voxel_grid.shape[1] // 2
    center_x = voxel_grid.shape[2] // 2
    
    fig = go.Figure()
    
    # XY slice (constant Z)
    fig.add_trace(go.Surface(
        z=[[center_z, center_z], [center_z, center_z]],
        x=[[0, voxel_grid.shape[2]], [0, voxel_grid.shape[2]]],
        y=[[0, 0], [voxel_grid.shape[1], voxel_grid.shape[1]]],
        surfacecolor=voxel_grid[center_z, :, :],
        colorscale=colorscale,
        showscale=True,
        name='XY Slice'
    ))
    
    # XZ slice (constant Y)
    fig.add_trace(go.Surface(
        z=[[0, voxel_grid.shape[0]], [0, voxel_grid.shape[0]]],
        x=[[0, voxel_grid.shape[2]], [0, voxel_grid.shape[2]]],
        y=[[center_y, center_y], [center_y, center_y]],
        surfacecolor=voxel_grid[:, center_y, :],
        colorscale=colorscale,
        showscale=False,
        name='XZ Slice'
    ))
    
    # YZ slice (constant X)
    fig.add_trace(go.Surface(
        z=[[0, voxel_grid.shape[0]], [0, voxel_grid.shape[0]]],
        x=[[center_x, center_x], [center_x, center_x]],
        y=[[0, voxel_grid.shape[1]], [0, voxel_grid.shape[1]]],
        surfacecolor=voxel_grid[:, :, center_x],
        colorscale=colorscale,
        showscale=False,
        name='YZ Slice'
    ))
    
    fig.update_layout(
        title='3D Tomogram Orthogonal Slices',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True
    )
    
    fig.show()

def plot_tomogram_slices_2d(filename, slice_type='middle'):
    """
    Create 2D slice plots of the tomogram.
    
    Args:
        filename: path to TIFF tomogram file
        slice_type: 'middle', 'all', or specific slice number
    """
    # Load the tomogram
    print(f"Loading tomogram: {filename}")
    voxel_grid = load_tomogram_tiff(filename)
    
    if slice_type == 'middle':
        # Show middle slices in each direction
        fig = go.Figure()
        
        # Middle XY slice
        z_mid = voxel_grid.shape[0] // 2
        fig.add_trace(go.Heatmap(
            z=voxel_grid[z_mid, :, :],
            colorscale='Viridis',
            name=f'XY Slice (Z={z_mid})',
            showscale=True
        ))
        
        fig.update_layout(
            title=f'Middle XY Slice (Z={z_mid})',
            xaxis_title='X',
            yaxis_title='Y'
        )
        fig.show()
        
        # Middle XZ slice
        y_mid = voxel_grid.shape[1] // 2
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=voxel_grid[:, y_mid, :],
            colorscale='Viridis',
            name=f'XZ Slice (Y={y_mid})',
            showscale=True
        ))
        
        fig.update_layout(
            title=f'Middle XZ Slice (Y={y_mid})',
            xaxis_title='X',
            yaxis_title='Z'
        )
        fig.show()
        
        # Middle YZ slice
        x_mid = voxel_grid.shape[2] // 2
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=voxel_grid[:, :, x_mid],
            colorscale='Viridis',
            name=f'YZ Slice (X={x_mid})',
            showscale=True
        ))
        
        fig.update_layout(
            title=f'Middle YZ Slice (X={x_mid})',
            xaxis_title='Y',
            yaxis_title='Z'
        )
        fig.show()
        
    elif slice_type == 'all':
        # Show all slices in a grid
        n_slices = min(9, voxel_grid.shape[0])  # Show up to 9 slices
        slice_indices = np.linspace(0, voxel_grid.shape[0]-1, n_slices, dtype=int)
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[f'Z={i}' for i in slice_indices],
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        for i, z_idx in enumerate(slice_indices):
            row = i // 3 + 1
            col = i % 3 + 1
            
            fig.add_trace(
                go.Heatmap(
                    z=voxel_grid[z_idx, :, :],
                    colorscale='Viridis',
                    showscale=(i == 0)  # Only show colorbar for first plot
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='XY Slices Through Tomogram',
            height=800,
            width=800,
            showlegend=False
        )
        fig.show()

def main():
    """Main function to demonstrate tomogram utilities."""
    
    print("="*60)
    print("TOMogram Utilities")
    print("="*60)
    
    # Example usage would go here
    print("This module provides utilities for:")
    print("- Creating 3D tomograms from particle assemblies")
    print("- Saving/loading tomograms as TIFF files")
    print("- Visualizing tomograms in 3D")
    print("- Creating 2D slice views")
    
    print("\n" + "="*60)
    print("TOMogram Utilities Complete")
    print("="*60)

if __name__ == "__main__":
    main() 
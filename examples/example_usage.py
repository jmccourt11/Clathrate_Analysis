#%%
"""
Example Usage Script

This script demonstrates how to use the cleaned particle visualization modules.

Author: [Your Name]
Date: [Current Date]
"""

# Import the cleaned modules
import importlib
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Reload the modules
import clathrate_analysis
import truncation_analysis
import tomogram_utils

importlib.reload(clathrate_analysis)
importlib.reload(truncation_analysis) 
importlib.reload(tomogram_utils)

# Import specific functions after reload
from clathrate_analysis import (
    parse_particles_and_shape, plot_particles, voxelize_particles,
    create_2d_projections, plot_projections_with_ffts, plot_fft_magnitude,
    plot_clathrate_cavities, analyze_clathrate_structure, plot_cavity_analysis,
    duplicate_unit_cell, get_bipyramid_geometry, plot_cavity_spheres,
    get_truncated_bipyramid_geometry
)

from truncation_analysis import (
    analyze_truncation_effects, plot_truncation_comparison,
    plot_particles_with_truncation, plot_truncation_cavity_comparison
)

from tomogram_utils import (
    create_tomogram_from_particles, plot_3d_tomogram
)

import numpy as np
import plotly.graph_objects as go

def get_cube_geometry(center, size=0.2):
    a = size / 2
    vertices = np.array([
        [-a, -a, -a],
        [+a, -a, -a],
        [+a, +a, -a],
        [-a, +a, -a],
        [-a, -a, +a],
        [+a, -a, +a],
        [+a, +a, +a],
        [-a, +a, +a],
    ]) + np.array(center)
    faces = [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [1, 2, 6], [1, 6, 5],  # right
        [0, 3, 7], [0, 7, 4],  # left
    ]
    return vertices, faces

# Load the file
filename = 'C:\\Users\\b304014\\Software\\blee\\models\\Cages\\Cage A new.pos'

# Parse particles and shape data
particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(filename)

# Set truncation factor
truncation_factor = 0.5  # Can be adjusted between 0.0 and 1.0

# Improved cavity detection parameters
grid_size = 128  # Increased from 64 for better resolution
min_cavity_size = 20  # Reduced to detect smaller cavities
padding = 0.15  # Moderate padding for better cavity detection

# # For 2x2x2 duplication
# grid_size = 128*2
# min_cavity_size = 100 # Number of voxels for a cavity to be considered

print(f"Minimum cavity size: {min_cavity_size}")
print(f"Grid size: {grid_size}")
print(f"Truncation factor: {truncation_factor}")
print(f"Padding: {padding}")

# Get truncated shape faces for tomogram creation
_, shape_faces, _ = get_truncated_bipyramid_geometry(
    shape_vertices, 
    shape_color,
    truncation_factor=truncation_factor
)

# Create nxnxn duplicated particles
duplicated_particles = duplicate_unit_cell(particles, nx=1, ny=1, nz=1, simulation_data=simulation_data)
duplicated_positions = [(pos, quat) for pos, quat, _ in duplicated_particles]

# Plot the duplicated truncated particles with cavities
print(f"\nPlotting duplicated particles with truncation factor {truncation_factor} and cavities...")
cavity_centers, cavity_volumes, cavity_radii = plot_clathrate_cavities(
    particles=duplicated_positions,
    shape_vertices=shape_vertices, 
    shape_color=shape_color,
    geometry_func=get_truncated_bipyramid_geometry,
    truncation_factor=truncation_factor,
    show_particles=True,
    show_cavities=True,
    show_spheres=True,
    grid_size=grid_size,
    padding=padding,  # Use the improved padding value
    simulation_data=simulation_data,
    min_cavity_size=min_cavity_size,
    keep_largest_cavity_only=False  # Show all cavities first
)

print(f"Found {len(cavity_centers)} cavities")
if len(cavity_radii) > 0:
    print(f"Average cavity radius: {np.mean(cavity_radii):.3f}")
    print(f"Largest cavity radius: {np.max(cavity_radii):.3f}")
    print(f"Largest cavity volume: {np.max(cavity_volumes):.6f}")
    
    # Print details for each cavity
    for i, (center, volume, radius) in enumerate(zip(cavity_centers, cavity_volumes, cavity_radii)):
        print(f"Cavity {i+1}: center={center}, volume={volume:.6f}, radius={radius:.6f}")
else:
    print("No cavities detected. This might indicate:")
    print("1. The particles are too close together")
    print("2. The truncation factor is too high")
    print("3. The grid resolution is too low")
    print("4. The minimum cavity size threshold is too high")




#%%
# Plot a cube at the first cavity center (if any)
if len(cavity_centers) > 0:
    # Set cube size to fit inside the cavity's inscribed sphere
    cube_size = 2 * cavity_radii[0]
    cube_vertices, cube_faces = get_cube_geometry(cavity_centers[0], size=cube_size)
    x, y, z = cube_vertices[:,0], cube_vertices[:,1], cube_vertices[:,2]
    i, j, k = zip(*cube_faces)
    fig = go.Figure()
    # Add the cube
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='green',
        opacity=0.7,
        name='Cube in Cavity'
    ))
    # Optionally, add the particles as transparent mesh for context
    vertices, faces, default_color = get_truncated_bipyramid_geometry(shape_vertices, shape_color, truncation_factor)
    for pos, quat in duplicated_positions:
        r = clathrate_analysis.R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        verts_rot = r.apply(vertices) + np.array(pos)
        x_p, y_p, z_p = verts_rot[:,0], verts_rot[:,1], verts_rot[:,2]
        i_p, j_p, k_p = zip(*faces)
        fig.add_trace(go.Mesh3d(
            x=x_p, y=y_p, z=z_p,
            i=i_p, j=j_p, k=k_p,
            color=default_color,
            opacity=0.2,
            name='Particles'
        ))
    fig.update_layout(
        title='Cube Inserted at First Cavity Center',
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



# Compute and plot 3D FFT of the duplicated particle assembly
print("\nComputing and plotting 3D FFT of the duplicated particle assembly...")
voxel_grid, edges = voxelize_particles(
    duplicated_positions,
    grid_size=grid_size,
    shape_vertices=shape_vertices,
    shape_faces=shape_faces
)
plot_fft_magnitude(voxel_grid, edges, log_scale=True, threshold=0.65)

# Create a new particle list with a cube at the first cavity center
# The cube is represented as (position, quaternion), where quaternion is identity (no rotation)
duplicated_positions_with_cube = list(duplicated_positions)
if len(cavity_centers) > 0:
    cube_pos = tuple(cavity_centers[0])
    cube_quat = (1, 0, 0, 0)  # Identity quaternion
    duplicated_positions_with_cube.append((cube_pos, cube_quat))

# Helper function to voxelize a mixed assembly (particles + cube)
def voxelize_mixed_particles(particles, shape_vertices, shape_faces, cube_vertices, cube_faces, grid_size):
    from clathrate_analysis import R
    # Assume the last particle is the cube
    n = len(particles)
    positions = np.array([pos for pos, _ in particles])
    min_corner = positions.min(axis=0)
    max_corner = positions.max(axis=0)
    box_size = max_corner - min_corner
    min_corner = min_corner - 0.1 * box_size
    max_corner = max_corner + 0.1 * box_size
    edges = [
        np.linspace(min_corner[d], max_corner[d], grid_size + 1)
        for d in range(3)
    ]
    centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    for idx, (pos, quat) in enumerate(particles):
        if idx == n - 1:  # last is cube
            verts = cube_vertices
            faces = cube_faces
        else:
            verts = shape_vertices
            faces = shape_faces
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        verts_rot = r.apply(verts) + np.array(pos)
        from scipy.spatial import ConvexHull
        hull = ConvexHull(verts_rot)
        eqs = hull.equations
        minv = verts_rot.min(axis=0)
        maxv = verts_rot.max(axis=0)
        idx_min = [np.searchsorted(centers[d], minv[d], side='left') for d in range(3)]
        idx_max = [np.searchsorted(centers[d], maxv[d], side='right') for d in range(3)]
        idx_min = [max(0, idx_min[d]) for d in range(3)]
        idx_max = [min(grid_size, idx_max[d]) for d in range(3)]
        xs = centers[0][idx_min[0]:idx_max[0]]
        ys = centers[1][idx_min[1]:idx_max[1]]
        zs = centers[2][idx_min[2]:idx_max[2]]
        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
        pts = np.stack([XX, YY, ZZ], axis=-1).reshape(-1, 3)
        inside = np.all((eqs[:,:3] @ pts.T + eqs[:,3:]) <= 1e-8, axis=0)
        idxs = np.argwhere(inside)
        for flat_idx in idxs:
            i = flat_idx[0] // (len(ys)*len(zs))
            j = (flat_idx[0] // len(zs)) % len(ys)
            k = flat_idx[0] % len(zs)
            voxel_grid[idx_min[0]+i, idx_min[1]+j, idx_min[2]+k] = 1.0
    return voxel_grid, edges

# Compute and plot 3D FFT of the duplicated particle assembly WITH cube in cavity
if len(cavity_centers) > 0:
    print("\nComputing and plotting 3D FFT of the duplicated particle assembly WITH cube in cavity...")
    voxel_grid_with_cube, edges_with_cube = voxelize_mixed_particles(
        duplicated_positions_with_cube,
        shape_vertices, shape_faces,
        cube_vertices, cube_faces,
        grid_size=grid_size
    )
    plot_fft_magnitude(voxel_grid_with_cube, edges_with_cube, log_scale=True, threshold=0.65)




# %%

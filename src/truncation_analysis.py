#%%
"""
Truncation Analysis Module

This module provides functions for analyzing truncated triangular bipyramids,
including volume/surface area calculations and visualization.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from clathrate_analysis import get_truncated_bipyramid_geometry, duplicate_unit_cell, get_mesh_trace, get_cell_color
import matplotlib.pyplot as plt
import pandas as pd
from clathrate_analysis import detect_clathrate_cavities, get_truncated_bipyramid_geometry

def get_truncated_bipyramid_volume_surface(truncation_factor):
    """
    Calculate volume and surface area of truncated triangular bipyramid.
    
    Based on analytical formulas:
    V(t) = V_p * (1 - 3/8 * t^3)
    S(t) = S_p * [1 - 1/6 * (6-√6) * t^2]
    
    where V_p = 18 and S_p = 27√3 are the volume and surface area of the perfect TBP.
    
    Args:
        truncation_factor: truncation parameter t (0-1)
    
    Returns:
        volume: volume of truncated TBP
        surface_area: surface area of truncated TBP
    """
    t = truncation_factor
    
    # Perfect TBP values
    V_p = 18
    S_p = 27 * np.sqrt(3)
    
    # Volume formula: V(t) = V_p * (1 - 3/8 * t^3)
    volume = V_p * (1 - (3/8) * t**3)
    
    # Surface area formula: S(t) = S_p * [1 - 1/6 * (6-√6) * t^2]
    surface_area = S_p * (1 - (1/6) * (6 - np.sqrt(6)) * t**2)
    
    return volume, surface_area

def analyze_truncation_effects(truncation_factors=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    """
    Analyze how truncation affects volume and surface area.
    
    Args:
        truncation_factors: list of truncation parameters to analyze
    """
    print("Truncation Analysis:")
    print("=" * 50)
    print(f"{'t':<6} {'Volume':<12} {'Surface Area':<15} {'V/V₀':<8} {'S/S₀':<8}")
    print("-" * 50)
    
    for t in truncation_factors:
        volume, surface_area = get_truncated_bipyramid_volume_surface(t)
        V_ratio = volume / 18  # V/V₀
        S_ratio = surface_area / (27 * np.sqrt(3))  # S/S₀
        
        print(f"{t:<6.1f} {volume:<12.3f} {surface_area:<15.3f} {V_ratio:<8.3f} {S_ratio:<8.3f}")
    
    print("\nNote: V₀ = 18, S₀ = 27√3 ≈ 46.765")

def plot_truncation_comparison(shape_vertices=None, shape_color=None, truncation_factors=[0.0, 0.3, 0.6, 1.0]):
    """
    Plot comparison of different truncation levels for a single particle.
    
    Args:
        shape_vertices: numpy array of vertex coordinates
        shape_color: color string
        truncation_factors: list of truncation parameters to compare
    """

    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f't = {t}' for t in truncation_factors],
        specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}],
               [{'type': 'mesh3d'}, {'type': 'mesh3d'}]]
    )
    
    # Use first particle position and orientation for comparison
    pos = (0, 0, 0)  # Center position
    quat = (1, 0, 0, 0)  # No rotation
    
    for i, t in enumerate(truncation_factors):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # Get truncated geometry
        vertices, faces, color = get_truncated_bipyramid_geometry(shape_vertices, shape_color, t)
        
        # Apply rotation and translation
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        verts_rot = r.apply(vertices) + np.array(pos)
        
        # Create mesh trace
        x, y, z = verts_rot[:, 0], verts_rot[:, 1], verts_rot[:, 2]
        i_faces, j_faces, k_faces = zip(*faces)
        
        fig.add_trace(
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i_faces, j=j_faces, k=k_faces,
                opacity=0.8,
                color=color,
                flatshading=True,
                showscale=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Truncated Triangular Bipyramid Comparison',
        height=800,
        width=800,
        showlegend=False
    )
    
    fig.show()

def plot_truncated_particle(pos, quat, shape_vertices=None, shape_color=None, 
                           truncation_factor=0.3, plot_type='surface', show_axes=True):
    """
    Plot a single truncated particle to visualize its geometry.
    
    Args:
        pos: (x, y, z) position tuple
        quat: (qw, qx, qy, qz) quaternion tuple
        shape_vertices: numpy array of vertex coordinates
        shape_color: color string
        truncation_factor: fraction to truncate at each vertex (0-1)
        plot_type: 'surface', 'wireframe', or 'both'
        show_axes: whether to show coordinate axes
    """
    
    vertices, faces, default_color = get_truncated_bipyramid_geometry(shape_vertices, shape_color, truncation_factor)
    
    # Apply rotation and translation
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    verts_rot = r.apply(vertices) + np.array(pos)
    
    fig = go.Figure()
    
    if plot_type in ['surface', 'both']:
        # Create surface mesh
        x, y, z = verts_rot[:, 0], verts_rot[:, 1], verts_rot[:, 2]
        i, j, k = zip(*faces)
        
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.8,
            color=default_color,
            flatshading=True,
            showscale=False,
            name='Surface'
        ))
    
    if plot_type in ['wireframe', 'both']:
        # Create wireframe edges
        edges = []
        for face in faces:
            for edge_idx in range(len(face)):
                start = face[edge_idx]
                end = face[(edge_idx + 1) % len(face)]
                edges.append((start, end))
        
        # Remove duplicate edges
        unique_edges = list(set(tuple(sorted(edge)) for edge in edges))
        
        for start, end in unique_edges:
            fig.add_trace(go.Scatter3d(
                x=[verts_rot[start, 0], verts_rot[end, 0]],
                y=[verts_rot[start, 1], verts_rot[end, 1]],
                z=[verts_rot[start, 2], verts_rot[end, 2]],
                mode='lines',
                line=dict(color='black', width=3),
                showlegend=False,
                name='Wireframe'
            ))
    
    # Add vertices as points
    fig.add_trace(go.Scatter3d(
        x=verts_rot[:, 0],
        y=verts_rot[:, 1],
        z=verts_rot[:, 2],
        mode='markers',
        marker=dict(size=6, color='red'),
        text=[f'V{i}' for i in range(len(verts_rot))],
        name='Vertices'
    ))
    
    if show_axes:
        # Add coordinate axes at particle center
        axis_length = 0.3
        center = np.array(pos)
        
        # X-axis (red)
        fig.add_trace(go.Scatter3d(
            x=[center[0], center[0] + axis_length],
            y=[center[1], center[1]],
            z=[center[2], center[2]],
            mode='lines',
            line=dict(color='red', width=5),
            name='X-axis'
        ))
        
        # Y-axis (green)
        fig.add_trace(go.Scatter3d(
            x=[center[0], center[0]],
            y=[center[1], center[1] + axis_length],
            z=[center[2], center[2]],
            mode='lines',
            line=dict(color='green', width=5),
            name='Y-axis'
        ))
        
        # Z-axis (blue)
        fig.add_trace(go.Scatter3d(
            x=[center[0], center[0]],
            y=[center[1], center[1]],
            z=[center[2], center[2] + axis_length],
            mode='lines',
            line=dict(color='blue', width=5),
            name='Z-axis'
        ))
    
    fig.update_layout(
        title=f'Truncated Particle (truncation={truncation_factor}) at Position {pos}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True
    )
    
    fig.show()

def plot_particles_with_truncation(particles, shape_vertices=None, shape_color=None, truncation_factor=0.3, 
                                 nx=1, ny=1, nz=1, show_duplicated=True, color_by_cell=False, simulation_data=None):
    """
    Plot particles with truncation option.
    
    Args:
        particles: List of (position, quaternion) tuples
        shape_vertices: numpy array of vertex coordinates from file
        shape_color: color string from file
        truncation_factor: fraction to truncate at each vertex (0-1)
        nx, ny, nz: Number of repetitions in each direction
        show_duplicated: Whether to show duplicated cells or just the original
        color_by_cell: If True, color each cell differently
        simulation_data: Dictionary containing simulation metadata
    """

    vertices, faces, default_color = get_truncated_bipyramid_geometry(shape_vertices, shape_color, truncation_factor)
    fig = go.Figure()

    if show_duplicated and (nx > 1 or ny > 1 or nz > 1):
        # Duplicate the unit cell
        duplicated_particles = duplicate_unit_cell(particles, nx, ny, nz, simulation_data)
        
        for pos, quat, (ix, iy, iz) in duplicated_particles:
            if color_by_cell:
                color = get_cell_color(ix, iy, iz, nx, ny, nz)
            else:
                color = default_color
            fig.add_trace(get_mesh_trace(pos, quat, vertices, faces, color))
    else:
        # Show only original particles
        for pos, quat in particles:
            fig.add_trace(get_mesh_trace(pos, quat, vertices, faces, default_color))

    # Update title based on duplication and truncation
    if show_duplicated and (nx > 1 or ny > 1 or nz > 1):
        title = f'Truncated Triangular Bipyramids (truncation={truncation_factor}) - {nx}×{ny}×{nz} Unit Cells'
    else:
        title = f'Truncated Triangular Bipyramids (truncation={truncation_factor}) - Single Unit Cell'

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
        ),
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False
    )

    fig.show()

def plot_saxs_comparison(base_path, truncation_factors=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    """
    Plot SAXS profiles for different truncation values.
    
    Args:
        base_path: Base path for SAXS data files
        truncation_factors: List of truncation parameters to plot
    """
    
    plt.figure(figsize=(10,6))

    for t in truncation_factors:
        # Read data with space separator and custom column names
        filename = f'{base_path}/truncated_t{t}_bipyramid_tomogram_spherical_avg.dat'
        try:
            data = pd.read_csv(
                filename,
                delim_whitespace=True,
                skiprows=1,
                names=['q', 'Calculated']
            )
            
            plt.plot(data['q'], data['Calculated'], label=f't={t}')
        except FileNotFoundError:
            print(f"Warning: File {filename} not found, skipping...")

    plt.xlabel('q')
    plt.xlim(0,3.0)
    plt.ylabel('Calculated')
    plt.title('SAXS Profiles for Different Truncation Values')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()


def plot_saxs_comparison_cavities_cubes(base_path, truncation_factor):
    """
    Plot SAXS profiles for different truncation values.
    
    Args:
        base_path: Base path for SAXS data files
        truncation_factors: List of truncation parameters to plot
    """
    plt.figure(figsize=(10,6))

    # Read data with space separator and custom column names
    filename = f'{base_path}/truncated_t{truncation_factor}_bipyramid_tomogram_spherical_avg.dat'
    try:
        data = pd.read_csv(
            filename,
            delim_whitespace=True,
            skiprows=1,
            names=['q', 'Calculated']
        )
        
        plt.plot(data['q'], data['Calculated'], label=f'Empty Cavities')
    except FileNotFoundError:
        print(f"Warning: File {filename} not found, skipping...")
        
    filename = f'{base_path}/truncated_t{truncation_factor}_bipyramid_tomogram_with_cubes_spherical_avg.dat'
    try:
        data = pd.read_csv(
            filename,
            delim_whitespace=True,
            skiprows=1,
            names=['q', 'Calculated']
        )
        
        plt.plot(data['q'], data['Calculated'], label=f'Cubic Cavities')
    except FileNotFoundError:
        print(f"Warning: File {filename} not found, skipping...")

    plt.xlabel('q')
    plt.xlim(0,3.0)
    plt.ylabel('Calculated')
    plt.title('SAXS Profiles with and without cubic cavities (truncation={truncation_factor})')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()


def plot_truncation_cavity_comparison(particles, shape_vertices=None, shape_color=None, 
                                     truncation_factors=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                     simulation_data=None):
    """
    Compare cavity properties across different truncation levels.
    
    Args:
        particles: List of (position, quaternion) tuples
        shape_vertices: numpy array of vertex coordinates from file
        shape_color: color string from file
        truncation_factors: list of truncation parameters to compare
        simulation_data: Dictionary containing simulation metadata
    """

    
    # Collect data for each truncation level
    cavity_counts = []
    total_volumes = []
    avg_radii = []
    max_radii = []
    
    print("\nTruncation-Cavity Analysis:")
    print("=" * 60)
    print(f"{'t':<6} {'Cavities':<10} {'Total Vol':<12} {'Avg Radius':<12} {'Max Radius':<12}")
    print("-" * 60)
    
    for t in truncation_factors:
        # Detect cavities for this truncation level
        cavity_voxels, cavity_centers, cavity_volumes, cavity_radii, _, _ = detect_clathrate_cavities(
            particles, shape_vertices, shape_color, 
            simulation_data=simulation_data,
            use_sphere_volume=True,
            geometry_func=get_truncated_bipyramid_geometry,
            truncation_factor=t
        )
        
        cavity_counts.append(len(cavity_centers))
        total_volumes.append(sum(cavity_volumes) if cavity_volumes else 0)
        avg_radii.append(np.mean(cavity_radii) if cavity_radii else 0)
        max_radii.append(max(cavity_radii) if cavity_radii else 0)
        
        print(f"{t:<6.1f} {len(cavity_centers):<10} {sum(cavity_volumes):<12.6f} "
              f"{np.mean(cavity_radii):<12.6f} {max(cavity_radii):<12.6f}")
    
    # Create comparison plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Number of Cavities vs Truncation',
            'Total Cavity Volume vs Truncation',
            'Average Cavity Radius vs Truncation',
            'Maximum Cavity Radius vs Truncation'
        ]
    )
    
    # Plot 1: Number of cavities
    fig.add_trace(go.Scatter(
        x=truncation_factors,
        y=cavity_counts,
        mode='lines+markers',
        name='Number of Cavities',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ), row=1, col=1)
    
    # Plot 2: Total volume
    fig.add_trace(go.Scatter(
        x=truncation_factors,
        y=total_volumes,
        mode='lines+markers',
        name='Total Cavity Volume',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ), row=1, col=2)
    
    # Plot 3: Average radius
    fig.add_trace(go.Scatter(
        x=truncation_factors,
        y=avg_radii,
        mode='lines+markers',
        name='Average Cavity Radius',
        line=dict(color='green', width=3),
        marker=dict(size=8)
    ), row=2, col=1)
    
    # Plot 4: Maximum radius
    fig.add_trace(go.Scatter(
        x=truncation_factors,
        y=max_radii,
        mode='lines+markers',
        name='Maximum Cavity Radius',
        line=dict(color='purple', width=3),
        marker=dict(size=8)
    ), row=2, col=2)
    
    fig.update_layout(
        title='Cavity Evolution with Truncation',
        height=800,
        width=1000,
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Truncation Factor (t)', row=1, col=1)
    fig.update_xaxes(title_text='Truncation Factor (t)', row=1, col=2)
    fig.update_xaxes(title_text='Truncation Factor (t)', row=2, col=1)
    fig.update_xaxes(title_text='Truncation Factor (t)', row=2, col=2)
    
    fig.update_yaxes(title_text='Number of Cavities', row=1, col=1)
    fig.update_yaxes(title_text='Total Volume', row=1, col=2)
    fig.update_yaxes(title_text='Average Radius', row=2, col=1)
    fig.update_yaxes(title_text='Maximum Radius', row=2, col=2)
    
    fig.show()
    
    return {
        'truncation_factors': truncation_factors,
        'cavity_counts': cavity_counts,
        'total_volumes': total_volumes,
        'avg_radii': avg_radii,
        'max_radii': max_radii
    }

def main():
    """Main function to demonstrate truncation analysis capabilities."""
    
    print("="*60)
    print("TRUNCATION ANALYSIS")
    print("="*60)
    
    # Analyze truncation effects using analytical formulas
    print("\n1. Analyzing truncation effects on volume and surface area...")
    analyze_truncation_effects(truncation_factors=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Example: Plot SAXS comparison (uncomment if you have the data files)
    # print("\n2. Plotting SAXS profiles for different truncation values...")
    # base_path = 'C:\\Users\\b304014\\Software\\blee\\models\\SAXS'
    # plot_saxs_comparison(base_path)
    
    print("\n" + "="*60)
    print("TRUNCATION ANALYSIS COMPLETE")
    print("="*60)
    

if __name__ == "__main__":
    main() 
# %%

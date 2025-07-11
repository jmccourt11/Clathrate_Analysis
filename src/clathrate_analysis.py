#%%
"""
Particle Visualization and Analysis Module

This module provides tools for visualizing and analyzing 3D particle assemblies,
particularly triangular bipyramids and clathrate structures.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from scipy import ndimage
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# ============================================================================
# DATA PARSING FUNCTIONS
# ============================================================================

def parse_shape_definition(shape_line):
    """
    Parse a shape definition line from file.
    
    Args:
        shape_line: String containing shape definition
        
    Returns:
        vertices: numpy array of vertex coordinates
        color: color string in rgba format
    """
    # Remove "shape" prefix and quotes if present
    if shape_line.startswith('shape '):
        shape_line = shape_line[6:]
    if shape_line.startswith('"') and shape_line.endswith('"'):
        shape_line = shape_line[1:-1]
    
    parts = shape_line.strip().split()
    
    if len(parts) < 2 or parts[0] != "poly3d":
        raise ValueError("Invalid shape definition format")
    
    num_vertices = int(parts[1])
    vertex_data = parts[2:2 + num_vertices * 3]
    
    if len(vertex_data) != num_vertices * 3:
        raise ValueError(f"Expected {num_vertices * 3} coordinate values, got {len(vertex_data)}")
    
    vertices = []
    for i in range(num_vertices):
        x = float(vertex_data[i * 3])
        y = float(vertex_data[i * 3 + 1])
        z = float(vertex_data[i * 3 + 2])
        vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    # Parse color (last part)
    color_hex = parts[-1]
    if len(color_hex) == 8:  # ARGB format
        a = int(color_hex[0:2], 16) / 255.0
        r = int(color_hex[2:4], 16)
        g = int(color_hex[4:6], 16)
        b = int(color_hex[6:8], 16)
        color = f'rgba({r},{g},{b},{a})'
    else:
        color = 'rgba(150,150,250,0.6)'  # Default color
    
    return vertices, color

def parse_particles_and_shape(filename):
    """
    Parse particle data and extract shape definition from file.
    
    Args:
        filename: Path to the particle data file
        
    Returns:
        particles: List of (position, quaternion) tuples
        shape_vertices: numpy array of vertex coordinates
        shape_color: color string in rgba format
        simulation_data: dict containing simulation metadata
    """
    particles = []
    shape_vertices = None
    shape_color = None
    simulation_data = {}
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments (except date)
            if line == '' or (line.startswith('#') and not line.startswith('#[data]')):
                continue
            
            # Parse date
            if line.startswith('//date:'):
                simulation_data['date'] = line[7:].strip()
                continue
            
            # Parse simulation statistics
            if line.startswith('#[data]'):
                simulation_data['data_columns'] = line[7:].strip().split('\t')
                continue
            
            # Parse simulation values
            if line_num == 3 and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 23:
                    simulation_data['simulation_values'] = {
                        'steps': int(parts[0]), 'time': float(parts[1]),
                        'volume': float(parts[2]), 'packing': float(parts[3]),
                        'pressure': float(parts[4]), 'msd': float(parts[5]),
                        'delta_x': float(parts[6]), 'delta_q': float(parts[7]),
                        'delta_v': float(parts[8]), 'accept_x': float(parts[9]),
                        'accept_q': float(parts[10]), 'accept_v': float(parts[11]),
                        'ensemble': int(parts[12]), 'shear': float(parts[13]),
                        'overlaps': int(parts[14]), 'x_length': float(parts[15]),
                        'y_length': float(parts[16]), 'z_length': float(parts[17]),
                        'xy_angle': float(parts[18]), 'xz_angle': float(parts[19]),
                        'yz_angle': float(parts[20]), 'rng_state': parts[21],
                        'rng_state_w': parts[22]
                    }
                continue
            
            # Parse translation
            if line.startswith('translation'):
                parts = line.split()
                if len(parts) == 4:
                    simulation_data['translation'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                continue
            
            # Parse zoom factor
            if line.startswith('zoomFactor'):
                parts = line.split()
                if len(parts) == 2:
                    simulation_data['zoom_factor'] = float(parts[1])
                continue
            
            # Parse box matrix
            if line.startswith('boxMatrix'):
                parts = line.split()
                if len(parts) == 10:
                    matrix_values = [float(x) for x in parts[1:]]
                    simulation_data['box_matrix'] = np.array(matrix_values).reshape(3, 3)
                continue
            
            # Parse shape definition
            if line.startswith('shape'):
                shape_rest = line[len('shape'):].lstrip()
                try:
                    shape_vertices, shape_color = parse_shape_definition(shape_rest)
                    print(f"Found shape definition: {len(shape_vertices)} vertices, color: {shape_color}")
                except Exception as e:
                    print(f"Warning: Could not parse shape definition: {e}")
                continue
            
            # Skip non-particle lines after line 9
            if line_num <= 9:
                continue
            
            # Parse particle data
            parts = line.split()
            if len(parts) < 7:
                continue
            
            try:
                floats = [float(x) for x in parts[:7]]
                x, y, z = floats[0:3]
                qw, qx, qy, qz = floats[3:7]
                particles.append(((x, y, z), (qw, qx, qy, qz)))
            except Exception:
                continue
    
    print(f"Found {len(particles)} particles")
    
    # Print simulation metadata summary
    if simulation_data:
        print("\nSimulation Metadata:")
        if 'date' in simulation_data:
            print(f"  Date: {simulation_data['date']}")
        if 'simulation_values' in simulation_data:
            sim_vals = simulation_data['simulation_values']
            print(f"  Steps: {sim_vals['steps']}")
            print(f"  Time: {sim_vals['time']}")
            print(f"  Volume: {sim_vals['volume']}")
            print(f"  Packing Fraction: {sim_vals['packing']}")
            print(f"  Box Dimensions: {sim_vals['x_length']:.3f} × {sim_vals['y_length']:.3f} × {sim_vals['z_length']:.3f}")
    
    return particles, shape_vertices, shape_color, simulation_data

# ============================================================================
# GEOMETRY FUNCTIONS
# ============================================================================

def get_bipyramid_geometry(shape_vertices=None, shape_color=None):
    """
    Get bipyramid geometry from parsed shape data or use default.
    
    Args:
        shape_vertices: numpy array of vertex coordinates from file
        shape_color: color string from file
    
    Returns:
        vertices: numpy array of vertex coordinates
        faces: list of face indices
        color: default color for the shape
    """
    if shape_vertices is not None:
        # Generate faces for triangular bipyramid (assuming 5 vertices)
        if len(shape_vertices) == 5:
            faces = [
                [3, 0, 1], [3, 1, 2], [3, 2, 0],  # Top apex to base
                [4, 0, 1], [4, 1, 2], [4, 2, 0]   # Bottom apex to base
            ]
        else:
            faces = []
        return shape_vertices, faces, shape_color or 'rgba(150,150,250,0.6)'
    
    # Default triangular bipyramid geometry
    top = [0, 0, 1]
    bottom = [0, 0, -1]
    base = [
        [np.cos(a), np.sin(a), 0]
        for a in np.linspace(0, 2 * np.pi, 4)[:-1]
    ]
    vertices = np.array([top] + base + [bottom]) * 0.2
    
    faces = [
        [0, 1, 2], [0, 2, 3], [0, 3, 1],  # Top apex to base
        [4, 1, 2], [4, 2, 3], [4, 3, 1]   # Bottom apex to base
    ]
    color = 'rgba(150,150,250,0.6)'
    return vertices, faces, color

def create_truncated_bipyramid(original_vertices, truncation_factor=0.3):
    """
    Create a truncated triangular bipyramid.
    
    Args:
        original_vertices: numpy array of original vertex coordinates (5 vertices)
        truncation_factor: truncation parameter t (0-1)
    
    Returns:
        vertices: numpy array of new vertex coordinates (18 vertices)
        faces: list of face indices (convex hull faces)
    """
    if len(original_vertices) != 5:
        raise ValueError("Original vertices must have exactly 5 vertices")
    
    # Map vertices: [V0=base1, V1=base2, V2=base3, V3=top_apex, V4=bottom_apex]
    o, p, q, r, s = original_vertices[3], original_vertices[0], original_vertices[1], original_vertices[2], original_vertices[4]
    t = truncation_factor
    
    # Create 18 new vertices by interpolating along edges
    new_vertices = []
    
    # Edge interpolations
    edges = [
        (o, p), (o, q), (o, r),  # Top apex to base
        (s, p), (s, q), (s, r),  # Bottom apex to base
        (q, r), (r, p), (p, q)   # Base edges
    ]
    
    for v1, v2 in edges:
        # Create two vertices along each edge
        new_vertices.extend([
            (1 - t/2) * v1 + (t/2) * v2,  # Closer to v1
            (1 - t/2) * v2 + (t/2) * v1   # Closer to v2
        ])
    
    new_vertices = np.array(new_vertices)
    
    # Use convex hull to find faces
    hull = ConvexHull(new_vertices)
    faces = [face.tolist() for face in hull.simplices]
    
    return new_vertices, faces

def get_truncated_bipyramid_geometry(shape_vertices=None, shape_color=None, truncation_factor=0.3):
    """
    Get truncated bipyramid geometry with specified truncation.
    
    Args:
        shape_vertices: numpy array of vertex coordinates from file
        shape_color: color string from file
        truncation_factor: fraction to truncate at each vertex (0-1)
    
    Returns:
        vertices: numpy array of vertex coordinates
        faces: list of face indices
        color: default color for the shape
    """
    if shape_vertices is not None and len(shape_vertices) == 5:
        vertices, faces = create_truncated_bipyramid(shape_vertices, truncation_factor)
        return vertices, faces, shape_color or 'rgba(150,150,250,0.6)'
    
    # Create default truncated bipyramid
    top = [0, 0, 1]
    bottom = [0, 0, -1]
    base = [
        [np.cos(a), np.sin(a), 0]
        for a in np.linspace(0, 2 * np.pi, 4)[:-1]
    ]
    default_vertices = np.array([base[0], base[1], base[2], top, bottom]) * 0.2
    
    vertices, faces = create_truncated_bipyramid(default_vertices, truncation_factor)
    color = 'rgba(150,150,250,0.6)'
    return vertices, faces, color

# ============================================================================
# UNIT CELL AND ASSEMBLY FUNCTIONS
# ============================================================================

def get_unit_cell_dimensions(particles, simulation_data=None):
    """
    Calculate unit cell dimensions from simulation metadata or particle positions.
    
    Args:
        particles: List of (position, quaternion) tuples
        simulation_data: Dictionary containing simulation metadata
    
    Returns:
        cell_dims: [x, y, z] dimensions of the unit cell
    """
    if simulation_data and 'simulation_values' in simulation_data:
        sim_vals = simulation_data['simulation_values']
        return [sim_vals['x_length'], sim_vals['y_length'], sim_vals['z_length']]
    
    if not particles:
        return [1.0, 1.0, 1.0]
    
    positions = np.array([pos for pos, _ in particles])
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    cell_dims = max_coords - min_coords
    return cell_dims

def duplicate_unit_cell(particles, nx=1, ny=1, nz=1, simulation_data=None):
    """
    Duplicate the unit cell nx, ny, nz times in x, y, z directions.
    
    Args:
        particles: List of (position, quaternion) tuples
        nx, ny, nz: Number of repetitions in each direction
        simulation_data: Dictionary containing simulation metadata
    
    Returns:
        List of duplicated particles with cell indices
    """
    if not particles:
        return []
    
    cell_dims = get_unit_cell_dimensions(particles, simulation_data)
    duplicated_particles = []
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                offset = np.array([ix * cell_dims[0], iy * cell_dims[1], iz * cell_dims[2]])
                
                for pos, quat in particles:
                    new_pos = tuple(np.array(pos) + offset)
                    duplicated_particles.append((new_pos, quat, (ix, iy, iz)))
    
    return duplicated_particles

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def get_mesh_trace(pos, quat, vertices, faces, color='rgba(150,150,250,0.6)'):
    """Generate mesh trace for a single particle."""
    if not faces:
        print("Warning: No faces provided for mesh trace. Skipping this particle.")
        return go.Scatter3d()
    
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    verts_rot = r.apply(vertices) + np.array(pos)
    x, y, z = verts_rot[:, 0], verts_rot[:, 1], verts_rot[:, 2]
    i, j, k = zip(*faces)
    
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.6,
        color=color,
        flatshading=True,
        showscale=False
    )

def get_cell_color(ix, iy, iz, nx, ny, nz):
    """Generate a color based on the cell position for visualization."""
    if nx == 1 and ny == 1 and nz == 1:
        return 'rgba(150,150,250,0.6)'
    
    r = int(100 + 155 * ix / max(1, nx-1))
    g = int(100 + 155 * iy / max(1, ny-1))
    b = int(100 + 155 * iz / max(1, nz-1))
    
    return f'rgba({r},{g},{b},0.6)'

def plot_particles(particles, nx=1, ny=1, nz=1, show_duplicated=True, 
                  shape_vertices=None, shape_color=None, color_by_cell=False, 
                  simulation_data=None, geometry_func=None, truncation_factor=None):
    """
    Plot particles with optional unit cell duplication.
    
    Args:
        particles: List of (position, quaternion) tuples
        nx, ny, nz: Number of repetitions in each direction
        show_duplicated: Whether to show duplicated cells or just the original
        shape_vertices: numpy array of vertex coordinates from file
        shape_color: color string from file
        color_by_cell: If True, color each cell differently
        simulation_data: Dictionary containing simulation metadata
        geometry_func: Function to get geometry
        truncation_factor: Truncation factor for geometry
    """
    if geometry_func is None:
        geometry_func = get_bipyramid_geometry
    
    if truncation_factor is not None and geometry_func == get_truncated_bipyramid_geometry:
        vertices, faces, default_color = geometry_func(shape_vertices, shape_color, truncation_factor)
    else:
        vertices, faces, default_color = geometry_func(shape_vertices, shape_color)
    
    fig = go.Figure()

    if show_duplicated and (nx > 1 or ny > 1 or nz > 1):
        duplicated_particles = duplicate_unit_cell(particles, nx, ny, nz, simulation_data)
        
        for pos, quat, (ix, iy, iz) in duplicated_particles:
            if color_by_cell:
                color = get_cell_color(ix, iy, iz, nx, ny, nz)
            else:
                color = default_color
            fig.add_trace(get_mesh_trace(pos, quat, vertices, faces, color))
    else:
        for pos, quat in particles:
            fig.add_trace(get_mesh_trace(pos, quat, vertices, faces, default_color))

    # Update title
    if show_duplicated and (nx > 1 or ny > 1 or nz > 1):
        title = f'3D Triangular Bipyramids - {nx}×{ny}×{nz} Unit Cells'
    else:
        title = '3D Triangular Bipyramids - Single Unit Cell'

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

# ============================================================================
# VOXELIZATION AND FFT FUNCTIONS
# ============================================================================

def voxelize_particles(particles, grid_size=64, padding=0.1, shape_vertices=None, shape_faces=None, geometry_func=None, truncation_factor=None):
    """
    Convert particle positions to a 3D voxel grid.
    
    Args:
        particles: list of (position, quaternion) tuples
        grid_size: number of voxels per dimension
        padding: fraction of box size to pad on each side
        shape_vertices: (N, 3) array of particle vertices
        shape_faces: list of face indices
        geometry_func: function to get geometry (default: None)
        truncation_factor: truncation parameter (default: None)
    Returns:
        voxel_grid: 3D numpy array
        grid_edges: (x_edges, y_edges, z_edges)
    """
    positions = np.array([pos for pos, _ in particles])
    min_corner = positions.min(axis=0)
    max_corner = positions.max(axis=0)
    box_size = max_corner - min_corner
    min_corner = min_corner - padding * box_size
    max_corner = max_corner + padding * box_size
    
    # Compute voxel edges and centers
    edges = [
        np.linspace(min_corner[d], max_corner[d], grid_size + 1)
        for d in range(3)
    ]
    centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    # For each particle, fill in the voxels inside the transformed shape
    for pos, quat in particles:
        # Use geometry_func if provided, else use shape_vertices
        if geometry_func is not None:
            if truncation_factor is not None:
                verts, _, _ = geometry_func(shape_vertices, shape_faces, truncation_factor)
            else:
                verts, _, _ = geometry_func(shape_vertices, shape_faces)
        else:
            verts = shape_vertices
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        verts = r.apply(verts) + np.array(pos)
        hull = ConvexHull(verts)
        eqs = hull.equations
        
        # Get bounding box in grid coordinates
        minv = verts.min(axis=0)
        maxv = verts.max(axis=0)
        idx_min = [np.searchsorted(centers[d], minv[d], side='left') for d in range(3)]
        idx_max = [np.searchsorted(centers[d], maxv[d], side='right') for d in range(3)]
        
        # Clamp to grid
        idx_min = [max(0, idx_min[d]) for d in range(3)]
        idx_max = [min(grid_size, idx_max[d]) for d in range(3)]
        
        # Generate all voxel centers in bounding box
        xs = centers[0][idx_min[0]:idx_max[0]]
        ys = centers[1][idx_min[1]:idx_max[1]]
        zs = centers[2][idx_min[2]:idx_max[2]]
        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
        pts = np.stack([XX, YY, ZZ], axis=-1).reshape(-1, 3)
        
        # Check all points at once
        inside = np.all((eqs[:,:3] @ pts.T + eqs[:,3:]) <= 1e-8, axis=0)
        
        # Set all inside voxels to 1
        idxs = np.argwhere(inside)
        for flat_idx in idxs:
            i = flat_idx[0] // (len(ys)*len(zs))
            j = (flat_idx[0] // len(zs)) % len(ys)
            k = flat_idx[0] % len(zs)
            voxel_grid[idx_min[0]+i, idx_min[1]+j, idx_min[2]+k] = 1.0
    
    return voxel_grid, edges

def create_2d_projections(voxel_grid, projection_type='sum'):
    """
    Create 2D projections of the 3D voxel grid.
    
    Args:
        voxel_grid: 3D numpy array
        projection_type: 'sum', 'max', or 'mean'
    
    Returns:
        projections: dict with keys 'xy', 'xz', 'yz'
    """
    if projection_type == 'sum':
        xy_proj = np.sum(voxel_grid, axis=2)
        xz_proj = np.sum(voxel_grid, axis=1)
        yz_proj = np.sum(voxel_grid, axis=0)
    elif projection_type == 'max':
        xy_proj = np.max(voxel_grid, axis=2)
        xz_proj = np.max(voxel_grid, axis=1)
        yz_proj = np.max(voxel_grid, axis=0)
    elif projection_type == 'mean':
        xy_proj = np.mean(voxel_grid, axis=2)
        xz_proj = np.mean(voxel_grid, axis=1)
        yz_proj = np.mean(voxel_grid, axis=0)
    else:
        raise ValueError("projection_type must be 'sum', 'max', or 'mean'")
    
    return {'xy': xy_proj, 'xz': xz_proj, 'yz': yz_proj}

def compute_2d_fft(projection):
    """
    Compute 2D FFT of a projection.
    
    Args:
        projection: 2D numpy array
    
    Returns:
        fft_mag: 2D array of FFT magnitude
        kx, ky: frequency arrays
    """
    fft_2d = np.fft.fft2(projection)
    fft_mag = np.abs(np.fft.fftshift(fft_2d))
    
    nx, ny = projection.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    
    return fft_mag, kx, ky

def plot_projections_with_ffts(projections, log_scale=True, threshold=0.1):
    """
    Plot original 2D projections alongside their corresponding FFTs.
    
    Args:
        projections: dict with 'xy', 'xz', 'yz' keys
        log_scale: whether to plot log10(magnitude) for FFTs
        threshold: fraction of max to threshold for FFT visualization
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'XY Projection', 'XZ Projection', 'YZ Projection',
            'XY Projection FFT', 'XZ Projection FFT', 'YZ Projection FFT'
        ],
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
               [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]]
    )
    
    for i, (plane, proj) in enumerate(projections.items()):
        # Plot original projection
        fig.add_trace(go.Heatmap(
            z=proj,
            colorscale='Viridis',
            showscale=True
        ), row=1, col=i+1)
        
        # Plot FFT
        fft_mag, kx, ky = compute_2d_fft(proj)
        
        if log_scale:
            fft_mag = np.log10(fft_mag + 1e-6)
        
        vmax = fft_mag.max()
        mask = fft_mag > (threshold * vmax)
        fft_mag[~mask] = 0
        
        fig.add_trace(go.Heatmap(
            z=fft_mag,
            x=kx,
            y=ky,
            colorscale='Viridis',
            showscale=True
        ), row=2, col=i+1)
    
    fig.update_layout(
        title='2D Projections and Their FFTs',
        height=800,
        width=1200,
        showlegend=False
    )
    fig.show()

def plot_fft_magnitude(voxel_grid, edges, log_scale=True, threshold=0.1):
    """
    Plot the magnitude of the 3D FFT of the voxel grid.
    
    Args:
        voxel_grid: 3D numpy array
        edges: (x_edges, y_edges, z_edges)
        log_scale: whether to plot log10(magnitude)
        threshold: fraction of max to threshold for visualization
    """
    fft_grid = np.fft.fftn(voxel_grid)
    fft_mag = np.abs(np.fft.fftshift(fft_grid))
    if log_scale:
        fft_mag = np.log10(fft_mag + 1e-6)
    
    # Threshold for visualization
    vmax = fft_mag.max()
    mask = fft_mag > (threshold * vmax)
    
    # Get coordinates
    grid_shape = fft_mag.shape
    kx = np.fft.fftshift(np.fft.fftfreq(grid_shape[0]))
    ky = np.fft.fftshift(np.fft.fftfreq(grid_shape[1]))
    kz = np.fft.fftshift(np.fft.fftfreq(grid_shape[2]))
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Only plot points above threshold
    x, y, z, val = KX[mask], KY[mask], KZ[mask], fft_mag[mask]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        mode='markers',
        marker=dict(
            size=3, 
            color=val, 
            colorscale='Viridis', 
            opacity=0.7,
            showscale=True,
            colorbar=dict(
                title='FFT Magnitude',
                x=1.1,
                len=0.8,
                thickness=20
            )
        ),
        text=[f"{v:.2f}" for v in val.flatten()]
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='kx', yaxis_title='ky', zaxis_title='kz',
            aspectmode='cube',
        ),
        title='3D FFT Magnitude (Structure Factor)',
        margin=dict(l=0, r=100, b=0, t=40),
        showlegend=False
    )
    fig.show()

# ============================================================================
# CLATHRATE CAVITY DETECTION
# ============================================================================

def calculate_cavity_sphere_radius(particles, cavity_center, shape_vertices=None, 
                                  max_radius=1.0, num_samples=100, simulation_data=None,
                                  geometry_func=None, truncation_factor=None):
    """
    Calculate the radius of the largest sphere that can fit at the cavity center.
    
    Args:
        particles: List of (position, quaternion) tuples
        cavity_center: [x, y, z] coordinates of cavity center
        shape_vertices: numpy array of vertex coordinates from file
        max_radius: maximum radius to test
        num_samples: number of radius samples to test
        simulation_data: Dictionary containing simulation metadata
        geometry_func: function to get geometry (default: None)
        truncation_factor: truncation parameter (default: None)
    
    Returns:
        max_fit_radius: radius of largest sphere that fits
        sphere_volume: volume of the sphere (4/3 * pi * r^3)
    """
    from scipy.spatial import ConvexHull
    
    # Test different radii
    radii = np.linspace(0.01, max_radius, num_samples)
    max_fit_radius = 0.0
    
    cavity_center = np.array(cavity_center)
    
    for radius in radii:
        # Check if sphere with this radius fits
        sphere_fits = True
        
        for pos, quat in particles:
            # Transform particle vertices to their position and orientation
            if geometry_func is not None and truncation_factor is not None:
                verts, _, _ = geometry_func(shape_vertices, None, truncation_factor)
            else:
                verts = shape_vertices
                
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            verts = r.apply(verts) + np.array(pos)
            
            # Calculate distance from cavity center to particle surface using convex hull
            hull = ConvexHull(verts)
            
            # Use the hull equations to calculate the signed distance
            # The equations are in the form ax + by + cz + d = 0
            # The signed distance is (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2)
            min_distance = float('inf')
            
            for eq in hull.equations:
                a, b, c, d = eq
                # Calculate signed distance from cavity center to this face
                signed_dist = (a * cavity_center[0] + b * cavity_center[1] + c * cavity_center[2] + d)
                # The absolute distance is the absolute value
                dist = abs(signed_dist) / np.sqrt(a**2 + b**2 + c**2)
                min_distance = min(min_distance, dist)
            
            # If sphere radius is larger than minimum distance, it doesn't fit
            if radius >= min_distance:
                sphere_fits = False
                break
        
        if sphere_fits:
            max_fit_radius = radius
        else:
            break
    
    # Calculate sphere volume
    sphere_volume = (4/3) * np.pi * max_fit_radius**3
    
    return max_fit_radius, sphere_volume

def detect_clathrate_cavities(particles, shape_vertices=None, shape_color=None, 
                             grid_size=64, padding=0.1, cavity_threshold=0.5, 
                             min_cavity_size=10, simulation_data=None, geometry_func=None, truncation_factor=None, keep_largest_cavity_only=False):
    """
    Detect clathrate cavities in a unit cell by finding empty spaces between particles.
    
    Args:
        particles: List of (position, quaternion) tuples
        shape_vertices: numpy array of vertex coordinates from file
        shape_color: color string from file
        grid_size: number of voxels per dimension for cavity detection
        padding: fraction of box size to pad on each side
        cavity_threshold: threshold for considering a region as cavity (0-1)
        min_cavity_size: minimum number of connected voxels to consider as cavity
        simulation_data: Dictionary containing simulation metadata
        geometry_func: function to get geometry (default: None)
        truncation_factor: truncation parameter (default: None)
        keep_largest_cavity_only: If True, only the largest cavity is kept (default: False)
    Returns:
        cavity_voxels: 3D numpy array with cavity regions marked
        cavity_centers: List of cavity center coordinates
        cavity_volumes: List of cavity volumes
        cavity_radii: List of cavity sphere radii
        voxel_grid: Original particle voxel grid
        edges: Grid edges for reference
    """
    # First, create the particle voxel grid
    print("Creating particle voxel grid...")
    voxel_grid, edges = voxelize_particles(particles, grid_size, padding, shape_vertices, None, geometry_func, truncation_factor)
    
    # Invert the grid: particles are 0, empty space is 1
    cavity_grid = 1 - voxel_grid
    
    print(f"Grid shape: {cavity_grid.shape}")
    print(f"Total empty voxels: {np.sum(cavity_grid)}")
    print(f"Total particle voxels: {np.sum(voxel_grid)}")
    
    # Apply minimal morphological operations to clean up noise
    print("Cleaning up cavity grid...")
    # Use very small structuring elements to preserve cavity structure
    struct_size = 1  # Use minimal structuring element
    print(f"Using structuring element size: {struct_size}")
    
    # Apply opening to remove very small noise (single voxels)
    cavity_grid = ndimage.binary_opening(cavity_grid, structure=np.ones((struct_size, struct_size, struct_size)))
    
    print(f"After opening - empty voxels: {np.sum(cavity_grid)}")
    
    # Find connected components (individual cavities)
    print("Finding connected cavity regions...")
    labeled_cavities, num_cavities = ndimage.label(cavity_grid)

    print(f"Found {num_cavities} total cavity regions")
    
    if num_cavities == 0:
        print("No cavities found!")
        return np.zeros_like(cavity_grid), [], [], [], voxel_grid, edges
    
    # Calculate sizes of all cavities
    sizes = ndimage.sum(cavity_grid, labeled_cavities, range(1, num_cavities + 1))
    print(f"Cavity sizes: {sizes}")

    # Sort cavities by size (largest first)
    sorted_indices = np.argsort(sizes)[::-1]
    
    cavity_centers = []
    cavity_volumes = []
    cavity_radii = []
    cavity_voxels = np.zeros_like(cavity_grid)
    
    # Process cavities starting from the largest
    for idx in sorted_indices:
        cavity_id = idx + 1  # labels start at 1
        cavity_mask = (labeled_cavities == cavity_id)
        cavity_size = np.sum(cavity_mask)
        
        print(f"Processing cavity {len(cavity_centers)+1} (ID {cavity_id}): size = {cavity_size}")
        
        if cavity_size >= min_cavity_size:
            # Find the optimal cavity center using distance transform
            from scipy.ndimage import distance_transform_edt
            
            # Compute distance transform within the cavity
            # This gives us the distance from each voxel to the nearest particle surface
            dist_transform = distance_transform_edt(cavity_mask)
            
            # Find the voxel with maximum distance (farthest from any particle)
            max_dist_idx = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
            max_distance = dist_transform[max_dist_idx]
            
            print(f"  Max distance from particles: {max_distance:.4f} voxels")
            
            # Convert to real coordinates
            centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
            real_center = np.array([
                centers[0][max_dist_idx[0]],
                centers[1][max_dist_idx[1]],
                centers[2][max_dist_idx[2]]
            ])
            
            # Calculate cavity volume (in voxel units)
            voxel_volume = np.prod([(edges[d][1] - edges[d][0]) for d in range(3)])
            cavity_volume = cavity_size * voxel_volume
            
            # Estimate radius from the distance transform result
            # The max_distance gives us the radius of the largest sphere that fits
            estimated_radius = max_distance * np.min([(edges[d][1] - edges[d][0]) for d in range(3)])
            
            # Fit the largest sphere at this optimal center
            print(f"Calculating sphere volume for cavity {len(cavity_centers)+1} at optimal center...")
            sphere_radius, sphere_volume = calculate_cavity_sphere_radius(
                particles, real_center, shape_vertices, simulation_data=simulation_data,
                geometry_func=geometry_func, truncation_factor=truncation_factor,
                max_radius=estimated_radius * 1.5  # Allow some margin for testing
            )
            
            cavity_centers.append(real_center)
            cavity_volumes.append(sphere_volume)  # Use sphere volume instead of voxel volume
            cavity_radii.append(sphere_radius)
            cavity_voxels[cavity_mask] = 1
            
            print(f"  Optimal center: {real_center}")
            print(f"  Voxel volume: {cavity_volume:.6f}")
            print(f"  Estimated radius from DT: {estimated_radius:.6f}")
            print(f"  Sphere volume: {sphere_volume:.6f}")
            print(f"  Sphere radius: {sphere_radius:.6f}")
            
            # If we only want the largest cavity, stop here
            if keep_largest_cavity_only:
                break
    
    print(f"Identified {len(cavity_centers)} significant cavities")
    for i, (center, volume, radius) in enumerate(zip(cavity_centers, cavity_volumes, cavity_radii)):
        print(f"  Cavity {i+1}: center={center}, volume={volume:.6f}, radius={radius:.6f}")
    
    return cavity_voxels, cavity_centers, cavity_volumes, cavity_radii, voxel_grid, edges

def plot_cavity_spheres(particles, cavity_centers, cavity_radii, shape_vertices=None, 
                       shape_color=None, show_particles=True, sphere_opacity=0.3,
                       simulation_data=None, geometry_func=None, truncation_factor=None):
    """
    Plot cavity centers as spheres with their calculated radii.
    
    Args:
        particles: List of (position, quaternion) tuples
        cavity_centers: List of cavity center coordinates
        cavity_radii: List of cavity sphere radii
        shape_vertices: numpy array of vertex coordinates from file
        shape_color: color string from file
        show_particles: Whether to show particles
        sphere_opacity: Opacity of cavity spheres
        simulation_data: Dictionary containing simulation metadata
        geometry_func: function to get geometry
        truncation_factor: passed to geometry_func if not None
    """
    if geometry_func is None:
        geometry_func = get_bipyramid_geometry
    
    fig = go.Figure()
    
    if show_particles:
        # Plot particles
        if truncation_factor is not None and geometry_func == get_truncated_bipyramid_geometry:
            vertices, faces, default_color = geometry_func(shape_vertices, shape_color, truncation_factor)
        else:
            vertices, faces, default_color = geometry_func(shape_vertices, shape_color)
        
        for pos, quat in particles:
            trace = get_mesh_trace(pos, quat, vertices, faces, default_color)
            if trace is not None:
                fig.add_trace(trace)
    
    # Plot cavity spheres
    for i, (center, radius) in enumerate(zip(cavity_centers, cavity_radii)):
        # Create sphere mesh
        phi = np.linspace(0, 2*np.pi, 20)
        theta = np.linspace(0, np.pi, 20)
        phi, theta = np.meshgrid(phi, theta)
        
        x = center[0] + radius * np.sin(theta) * np.cos(phi)
        y = center[1] + radius * np.sin(theta) * np.sin(phi)
        z = center[2] + radius * np.cos(theta)
        
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=sphere_opacity,
            colorscale='Blues',
            showscale=False,
            name=f'Cavity {i+1} Sphere (r={radius:.4f})'
        ))
    
    fig.update_layout(
        title=f'Cavity Spheres - {len(cavity_centers)} cavities',
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

def plot_clathrate_cavities(particles, shape_vertices=None, shape_color=None, 
                           cavity_threshold=0.5, min_cavity_size=10, grid_size=64, padding=0.1,
                           show_particles=True, show_cavities=True, show_spheres=False,
                           particle_opacity=0.6, cavity_opacity=0.8,
                           simulation_data=None, geometry_func=None, truncation_factor=None, keep_largest_cavity_only=False):
    """
    Plot clathrate assembly with detected cavities highlighted.
    """
    if geometry_func is None:
        geometry_func = get_bipyramid_geometry
    
    # Detect cavities
    cavity_voxels, cavity_centers, cavity_volumes, cavity_radii, particle_voxels, edges = detect_clathrate_cavities(
        particles, shape_vertices, shape_color, 
        grid_size=grid_size, padding=padding, 
        cavity_threshold=cavity_threshold, 
        min_cavity_size=min_cavity_size,
        simulation_data=simulation_data,
        geometry_func=geometry_func,
        truncation_factor=truncation_factor,
        keep_largest_cavity_only=keep_largest_cavity_only
    )
    
    if show_spheres:
        # Use the dedicated sphere visualization function
        plot_cavity_spheres(
            particles, cavity_centers, cavity_radii,
            shape_vertices, shape_color, show_particles,
            simulation_data=simulation_data, geometry_func=geometry_func,
            truncation_factor=truncation_factor
        )
        return cavity_centers, cavity_volumes, cavity_radii
    
    fig = go.Figure()
    
    if show_particles:
        # Plot particles
        if truncation_factor is not None and geometry_func == get_truncated_bipyramid_geometry:
            vertices, faces, default_color = geometry_func(shape_vertices, shape_color, truncation_factor)
        else:
            vertices, faces, default_color = geometry_func(shape_vertices, shape_color)
        
        for pos, quat in particles:
            trace = get_mesh_trace(pos, quat, vertices, faces, default_color)
            if trace is not None:
                fig.add_trace(trace)
    
    if show_cavities and len(cavity_centers) > 0:
        # Add cavity centers as spheres
        for i, center in enumerate(cavity_centers):
            fig.add_trace(go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode='markers',
                marker=dict(
                    size=16,
                    color='blue',
                    symbol='circle'
                ),
                text=[f'Cavity {i+1}<br>Volume: {cavity_volumes[i]:.6f}<br>Radius: {cavity_radii[i]:.6f}'],
                name=f'Cavity {i+1} Center'
            ))
    
    # Update layout
    title = f'Clathrate Assembly with {len(cavity_centers)} Cavities'
    if not show_particles:
        title += ' (Particles Hidden)'
    if not show_cavities:
        title += ' (Cavities Hidden)'
    
    fig.update_layout(
        title=title,
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
    
    return cavity_centers, cavity_volumes, cavity_radii

def analyze_clathrate_structure(particles, shape_vertices=None, shape_color=None, 
                               simulation_data=None, geometry_func=None, truncation_factor=None):
    """
    Analyze the clathrate structure and provide detailed information about cavities.
    
    Args:
        particles: List of (position, quaternion) tuples
        shape_vertices: numpy array of vertex coordinates from file
        shape_color: color string from file
        simulation_data: Dictionary containing simulation metadata
        geometry_func: function to get geometry (default: get_bipyramid_geometry)
        truncation_factor: passed to geometry_func if not None
    
    Returns:
        analysis_results: Dictionary containing analysis results
    """
    if geometry_func is None:
        geometry_func = get_bipyramid_geometry
    
    print("="*60)
    print("CLATHRATE STRUCTURE ANALYSIS")
    print("="*60)
    
    print(f"Number of particles: {len(particles)}")
    
    cell_dims = get_unit_cell_dimensions(particles, simulation_data)
    cell_volume = np.prod(cell_dims)
    print(f"Unit cell dimensions: {cell_dims}")
    print(f"Unit cell volume: {cell_volume:.3f}")
    
    cavity_voxels, cavity_centers, cavity_volumes, cavity_radii, particle_voxels, edges = detect_clathrate_cavities(
        particles, shape_vertices, shape_color, simulation_data=simulation_data,
        geometry_func=geometry_func,
        truncation_factor=truncation_factor
    )
    
    total_cavity_volume = sum(cavity_volumes)
    cavity_fraction = total_cavity_volume / cell_volume
    
    print(f"\nCavity Analysis:")
    print(f"  Number of cavities: {len(cavity_centers)}")
    print(f"  Total cavity volume: {total_cavity_volume:.3f}")
    print(f"  Cavity fraction: {cavity_fraction:.3f} ({cavity_fraction*100:.1f}%)")
    
    if len(cavity_volumes) > 0:
        print(f"\nIndividual Cavity Details:")
        print(f"{'Cavity':<8} {'Volume':<12} {'Radius':<12} {'Fraction':<12} {'Center':<20}")
        print("-" * 70)
        for i, (center, volume, radius) in enumerate(zip(cavity_centers, cavity_volumes, cavity_radii)):
            fraction = volume / cell_volume
            print(f"{i+1:<8} {volume:<12.6f} {radius:<12.6f} {fraction:<12.6f} {str(center):<20}")
    
    particle_volume = np.sum(particle_voxels) * np.prod([(edges[d][1] - edges[d][0]) for d in range(3)])
    particle_fraction = particle_volume / cell_volume
    
    print(f"\nParticle Analysis:")
    print(f"  Total particle volume: {particle_volume:.3f}")
    print(f"  Particle fraction: {particle_fraction:.3f} ({particle_fraction*100:.1f}%)")
    
    total_fraction = particle_fraction + cavity_fraction
    print(f"\nVolume Conservation Check:")
    print(f"  Particle + Cavity fraction: {total_fraction:.3f}")
    print(f"  Remaining space: {1 - total_fraction:.3f}")
    
    return {
        'num_particles': len(particles),
        'cell_volume': cell_volume,
        'num_cavities': len(cavity_centers),
        'total_cavity_volume': total_cavity_volume,
        'cavity_fraction': cavity_fraction,
        'particle_volume': particle_volume,
        'particle_fraction': particle_fraction,
        'cavity_centers': cavity_centers,
        'cavity_volumes': cavity_volumes,
        'cavity_radii': cavity_radii
    }

def plot_cavity_analysis(particles, shape_vertices=None, shape_color=None, 
                        simulation_data=None, geometry_func=None, truncation_factor=None):
    """
    Create comprehensive visualization of clathrate cavities with analysis.
    
    Args:
        particles: List of (position, quaternion) tuples
        shape_vertices: numpy array of vertex coordinates from file
        shape_color: color string from file
        simulation_data: Dictionary containing simulation metadata
        geometry_func: function to get geometry (default: get_bipyramid_geometry)
        truncation_factor: passed to geometry_func if not None
    
    Returns:
        analysis_results: Dictionary containing analysis results
    """
    if geometry_func is None:
        geometry_func = get_bipyramid_geometry
    
    analysis = analyze_clathrate_structure(particles, shape_vertices, shape_color, simulation_data, geometry_func, truncation_factor)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Particles Only',
            'Cavities Only', 
            'Particles + Cavities',
            'Cavity Volume Distribution'
        ],
        specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}],
               [{'type': 'mesh3d'}, {'type': 'bar'}]]
    )
    
    if truncation_factor is not None and geometry_func == get_truncated_bipyramid_geometry:
        vertices, faces, default_color = geometry_func(shape_vertices, shape_color, truncation_factor)
    else:
        vertices, faces, default_color = geometry_func(shape_vertices, shape_color)
    
    # Plot particles only
    for pos, quat in particles:
        trace = get_mesh_trace(pos, quat, vertices, faces, default_color)
        if trace is not None:
            fig.add_trace(trace, row=1, col=1)
    
    # Detect cavities
    cavity_voxels, cavity_centers, cavity_volumes, cavity_radii, _, edges = detect_clathrate_cavities(
        particles, shape_vertices, shape_color, simulation_data=simulation_data,
        geometry_func=geometry_func,
        truncation_factor=truncation_factor
    )
    
    # Plot cavities only
    if len(cavity_centers) > 0:
        z, y, x = np.mgrid[0:cavity_voxels.shape[0], 0:cavity_voxels.shape[1], 0:cavity_voxels.shape[2]]
        centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
        x_real = centers[0][x]
        y_real = centers[1][y]
        z_real = centers[2][z]
        
        fig.add_trace(go.Isosurface(
            x=x_real.flatten(),
            y=y_real.flatten(),
            z=z_real.flatten(),
            value=cavity_voxels.flatten(),
            isomin=0.5,
            isomax=1.0,
            opacity=0.8,
            surface_count=1,
            colorscale='Reds',
            caps=dict(x_show=False, y_show=False, z_show=False),
            showlegend=False
        ), row=1, col=2)
    
    # Plot particles + cavities
    for pos, quat in particles:
        trace = get_mesh_trace(pos, quat, vertices, faces, default_color)
        if trace is not None:
            trace.opacity = 0.4
            fig.add_trace(trace, row=2, col=1)
    
    if len(cavity_centers) > 0:
        fig.add_trace(go.Isosurface(
            x=x_real.flatten(),
            y=y_real.flatten(),
            z=z_real.flatten(),
            value=cavity_voxels.flatten(),
            isomin=0.5,
            isomax=1.0,
            opacity=0.6,
            surface_count=1,
            colorscale='Reds',
            caps=dict(x_show=False, y_show=False, z_show=False),
            showlegend=False
        ), row=2, col=1)
    
    # Plot cavity volume distribution
    if len(cavity_volumes) > 0:
        fig.add_trace(go.Bar(
            x=[f'Cavity {i+1}' for i in range(len(cavity_volumes))],
            y=cavity_volumes,
            name='Cavity Volumes',
            marker_color='red'
        ), row=2, col=2)
    
    fig.update_layout(
        title=f'Clathrate Structure Analysis - {len(particles)} particles, {len(cavity_centers)} cavities',
        height=800,
        width=1200,
        showlegend=False
    )
    
    fig.show()
    
    return analysis

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate the particle visualization capabilities."""
    
    # Example file path - replace with your actual file
    filename = 'C:\\Users\\b304014\\Software\\blee\\models\\ClaS_bipyramid_averaged.pos'
    
    # Parse both particles and shape definition from the same file
    particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(filename)
    
    # Get faces for the shape
    _, shape_faces, _ = get_bipyramid_geometry(shape_vertices, shape_color)
    
    print("="*60)
    print("PARTICLE VISUALIZATION AND ANALYSIS")
    print("="*60)
    
    # Example 1: Plot single unit cell
    print("\n1. Plotting single unit cell...")
    plot_particles(particles, nx=1, ny=1, nz=1, show_duplicated=False, 
                   shape_vertices=shape_vertices, shape_color=shape_color, 
                   simulation_data=simulation_data)
    
    # Example 2: Plot duplicated cells
    print("\n2. Plotting 3x3x3 duplicated unit cells...")
    plot_particles(particles, nx=3, ny=3, nz=3, show_duplicated=True, 
                   shape_vertices=shape_vertices, shape_color=shape_color, 
                   simulation_data=simulation_data)
    
    # Example 3: FFT analysis
    print("\n3. Computing 3D FFT of the assembly...")
    all_particles = duplicate_unit_cell(particles, nx=1, ny=1, nz=1, simulation_data=simulation_data)
    all_positions = [(pos, quat) for pos, quat, _ in all_particles]
    voxel_grid, edges = voxelize_particles(all_positions, grid_size=128, 
                                          shape_vertices=shape_vertices, shape_faces=shape_faces)
    plot_fft_magnitude(voxel_grid, edges, log_scale=True, threshold=0.65)
    
    # Example 4: 2D Projection FFTs
    print("\n4. Computing 2D projection FFTs...")
    projections = create_2d_projections(voxel_grid, projection_type='sum')
    plot_projections_with_ffts(projections, log_scale=True, threshold=0.1)
    
    # Example 5: Clathrate cavity detection
    print("\n5. Detecting clathrate cavities...")
    cavity_centers, cavity_volumes, cavity_radii = plot_clathrate_cavities(
        particles=particles,
        shape_vertices=shape_vertices,
        shape_color=shape_color,
        show_particles=True,
        show_cavities=True,
        show_spheres=True, # Added show_spheres=True
        simulation_data=simulation_data
    )
    
    # Example 6: Detailed clathrate structure analysis
    print("\n6. Performing detailed clathrate structure analysis...")
    analysis_results = analyze_clathrate_structure(
        particles=particles,
        shape_vertices=shape_vertices,
        shape_color=shape_color,
        simulation_data=simulation_data
    )
    
    # Example 7: Comprehensive cavity analysis visualization
    print("\n7. Creating comprehensive cavity analysis visualization...")
    plot_cavity_analysis(
        particles=particles,
        shape_vertices=shape_vertices,
        shape_color=shape_color,
        simulation_data=simulation_data
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main() 
# %%

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

def plot_cavity_objects(particles, cavity_centers, cavity_radii, shape_vertices=None, 
                     shape_color=None, show_particles=True, object_opacity=0.3,
                     object_color='rgba(0,100,255,1)', simulation_data=None, 
                     geometry_func=None, truncation_factor=None,
                     cavity_object_type='cube', cavity_object_scale=1.0):
    """
    Plot objects centered at cavity locations with sizes based on cavity radii.
    
    Args:
        particles: List of (position, quaternion) tuples
        cavity_centers: List of cavity center coordinates
        cavity_radii: List of cavity radii
        shape_vertices: numpy array of vertex coordinates from file
        shape_color: color string from file
        show_particles: Whether to show original particles
        object_opacity: Opacity of cavity objects
        object_color: Color of the objects (default: blue)
        simulation_data: Dictionary containing simulation metadata
        geometry_func: function to get geometry for particles
        truncation_factor: passed to geometry_func if not None
        cavity_object_type: Type of object to place at cavities ('cube', 'bipyramid', etc.)
        cavity_object_scale: Scale factor to adjust object size relative to cavity radius
        
    Returns:
        voxel_grid: 3D numpy array with particles (1.0) and cavity objects (2.0)
        edges: Grid edges for reference
    """
    if geometry_func is None:
        geometry_func = get_bipyramid_geometry
    
    # First create the voxel grid
    grid_size = 128  # Standard grid size
    padding = 0.1    # Standard padding
    
    # Get particle positions and box dimensions
    positions = np.array([pos for pos, _ in particles])
    min_corner = positions.min(axis=0)
    max_corner = positions.max(axis=0)
    box_size = max_corner - min_corner
    
    # Add padding
    min_corner = min_corner - padding * box_size
    max_corner = max_corner + padding * box_size
    
    # Create grid edges and centers
    edges = [np.linspace(min_corner[d], max_corner[d], grid_size + 1) for d in range(3)]
    centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    # Get base geometry for particles
    if truncation_factor is not None and geometry_func == get_truncated_bipyramid_geometry:
        particle_vertices, particle_faces, _ = geometry_func(shape_vertices, shape_color, truncation_factor)
    else:
        particle_vertices, particle_faces, _ = geometry_func(shape_vertices, shape_color)
    
    # Fill in particles with value 1.0
    if show_particles:
        for pos, quat in particles:
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            verts = r.apply(particle_vertices) + np.array(pos)
            hull = ConvexHull(verts)
            
            # Get bounding box in grid coordinates
            minv = verts.min(axis=0)
            maxv = verts.max(axis=0)
            idx_min = [max(0, np.searchsorted(centers[d], minv[d], side='left')) for d in range(3)]
            idx_max = [min(grid_size, np.searchsorted(centers[d], maxv[d], side='right')) for d in range(3)]
            
            # Fill in particle voxels
            for i in range(idx_min[0], idx_max[0]):
                for j in range(idx_min[1], idx_max[1]):
                    for k in range(idx_min[2], idx_max[2]):
                        point = np.array([centers[0][i], centers[1][j], centers[2][k]])
                        inside = True
                        for eq in hull.equations:
                            if np.dot(eq[:3], point) + eq[3] > 1e-10:
                                inside = False
                                break
                        if inside:
                            voxel_grid[i,j,k] = 1.0
    
    if cavity_centers and cavity_radii is not None:
        # Fill in cavity objects with value 2.0
        for center, radius in zip(cavity_centers, cavity_radii):
            # Get object geometry based on type
            if cavity_object_type == 'cube':
                # For cubes, edge_length = 2 * radius / √3 to fit inside sphere
                edge_length = 2 * radius * cavity_object_scale / np.sqrt(3)
                vertices, faces = get_cube_geometry(center, size=edge_length)
            elif cavity_object_type == 'bipyramid':
                # For bipyramids, scale the original vertices to fit
                vertices = particle_vertices.copy() * (radius * cavity_object_scale)
                faces = particle_faces
                # Move to cavity center
                vertices = vertices + center
            else:
                raise ValueError(f"Unsupported cavity object type: {cavity_object_type}")
            
            # Create convex hull for the object
            hull = ConvexHull(vertices)
            
            # Get bounding box in grid coordinates
            minv = vertices.min(axis=0)
            maxv = vertices.max(axis=0)
            idx_min = [max(0, np.searchsorted(centers[d], minv[d], side='left')) for d in range(3)]
            idx_max = [min(grid_size, np.searchsorted(centers[d], maxv[d], side='right')) for d in range(3)]
            
            # Fill in object voxels
            for i in range(idx_min[0], idx_max[0]):
                for j in range(idx_min[1], idx_max[1]):
                    for k in range(idx_min[2], idx_max[2]):
                        point = np.array([centers[0][i], centers[1][j], centers[2][k]])
                        inside = True
                        for eq in hull.equations:
                            if np.dot(eq[:3], point) + eq[3] > 1e-10:
                                inside = False
                                break
                        if inside:
                            voxel_grid[i,j,k] = 2.0
    
    # Create visualization
    fig = go.Figure()
    
    if show_particles:
        # Plot original particles
        for pos, quat in particles:
            trace = get_mesh_trace(pos, quat, particle_vertices, particle_faces, shape_color or 'rgba(150,150,250,0.6)')
            if trace is not None:
                fig.add_trace(trace)
    
    # Plot cavity objects
    if cavity_centers and cavity_radii is not None:
        for cavity_idx, (center, radius) in enumerate(zip(cavity_centers, cavity_radii)):
            if cavity_object_type == 'cube':
                edge_length = 2 * radius * cavity_object_scale / np.sqrt(3)
                vertices, faces = get_cube_geometry(center, size=edge_length)
                obj_name = f'Cavity {cavity_idx+1} Cube (edge={edge_length:.4f}, cavity_radius={radius:.4f})'
            elif cavity_object_type == 'bipyramid':
                vertices = particle_vertices.copy() * (radius * cavity_object_scale) + center
                faces = particle_faces
                obj_name = f'Cavity {cavity_idx+1} Bipyramid (radius={radius:.4f})'
            
            x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            i_faces, j_faces, k_faces = zip(*faces) if faces else ([], [], [])
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i_faces, j=j_faces, k=k_faces,
                opacity=object_opacity,
                color=object_color,
                name=obj_name
            ))
        
        fig.update_layout(
            title=f'Cavity Objects ({cavity_object_type}) - {len(cavity_centers)} cavities',
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
    
    return voxel_grid, edges

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
def detect_simple_cavities(particles, shape_vertices=None, shape_color=None, 
                          grid_size=128, padding=0.2, geometry_func=None, truncation_factor=None,
                          min_radius=0.1, min_separation=0.3, boundary_margin=0.3,
                          min_surrounding_particles=6, max_empty_neighbors_fraction=0.4,
                          debug=True):  # Added debug parameter
    """
    A simplified version of cavity detection that focuses on core concepts.
    
    Args:
        particles: List of (position, quaternion) tuples
        shape_vertices: numpy array of vertex coordinates
        shape_color: color string
        grid_size: number of voxels per dimension
        padding: fraction of box size to pad
        geometry_func: function to get geometry
        truncation_factor: truncation parameter
        min_radius: minimum cavity radius to consider (in real units)
        min_separation: minimum separation between cavity centers (in real units)
        boundary_margin: minimum distance from box boundaries (in real units)
        min_surrounding_particles: minimum number of particles that should be near a cavity
        max_empty_neighbors_fraction: maximum fraction of neighboring voxels that can be empty
    """
    print("\nSimplified Cavity Detection:")
    print("============================")
    print(f"\nParameters:")
    print(f"  min_radius: {min_radius}")
    print(f"  min_separation: {min_separation}")
    print(f"  boundary_margin: {boundary_margin}")
    print(f"  min_surrounding_particles: {min_surrounding_particles}")
    print(f"  max_empty_neighbors_fraction: {max_empty_neighbors_fraction}")
    
    # 1. Create particle grid and determine actual structure boundaries
    print("\n1. Creating particle grid...")
    positions = np.array([pos for pos, _ in particles])
    min_corner = positions.min(axis=0)
    max_corner = positions.max(axis=0)
    box_size = max_corner - min_corner
    
    # Store actual structure boundaries before adding padding
    structure_min = min_corner
    structure_max = max_corner
    print(f"Structure boundaries:")
    print(f"  Min: {structure_min}")
    print(f"  Max: {structure_max}")
    
    # Add padding for grid computation
    min_corner = min_corner - padding * box_size
    max_corner = max_corner + padding * box_size
    
    # Create grid edges and centers
    edges = [np.linspace(min_corner[d], max_corner[d], grid_size + 1) for d in range(3)]
    centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    
    voxel_size = edges[0][1] - edges[0][0]
    print(f"Box size: {box_size}")
    print(f"Grid resolution: {voxel_size:.4f} units per voxel")
    
    # 2. Fill in particles
    print("\n2. Filling particle voxels...")
    
    # Get particle geometry
    if geometry_func is not None and truncation_factor is not None:
        vertices, _, _ = geometry_func(shape_vertices, None, truncation_factor)
    else:
        vertices = shape_vertices
    
    for pos, quat in particles:
        # Transform vertices
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        verts = r.apply(vertices) + np.array(pos)
        
        # Get bounding box in grid coordinates
        minv = verts.min(axis=0)
        maxv = verts.max(axis=0)
        idx_min = [max(0, np.searchsorted(centers[d], minv[d], side='left')) for d in range(3)]
        idx_max = [min(grid_size, np.searchsorted(centers[d], maxv[d], side='right')) for d in range(3)]
        
        # Create convex hull
        hull = ConvexHull(verts)
        
        # Check voxels in bounding box
        for i in range(idx_min[0], idx_max[0]):
            for j in range(idx_min[1], idx_max[1]):
                for k in range(idx_min[2], idx_max[2]):
                    point = np.array([centers[0][i], centers[1][j], centers[2][k]])
                    
                    # Check if point is inside hull
                    inside = True
                    for eq in hull.equations:
                        if np.dot(eq[:3], point) + eq[3] > 1e-10:
                            inside = False
                            break
                    
                    if inside:
                        voxel_grid[i,j,k] = True
    
    print(f"Filled voxels: {np.sum(voxel_grid)}")
    
    # 3. Find empty regions (cavities)
    print("\n3. Finding cavities...")
    cavity_grid = ~voxel_grid
    print(f"Initial empty voxels: {np.sum(cavity_grid)}")
    
    # Create a mask for the valid region (away from boundaries)
    valid_region = np.ones_like(cavity_grid, dtype=bool)
    for d in range(3):
        # Convert boundary_margin to grid coordinates
        margin_voxels = int(boundary_margin / voxel_size)
        
        # Find indices corresponding to structure boundaries
        min_idx = np.searchsorted(centers[d], structure_min[d] + boundary_margin)
        max_idx = np.searchsorted(centers[d], structure_max[d] - boundary_margin)
        
        # Create slices for this dimension
        slices = [slice(None)] * 3
        
        # Mask out regions before min boundary
        slices[d] = slice(0, min_idx)
        valid_region[tuple(slices)] = False
        
        # Mask out regions after max boundary
        slices[d] = slice(max_idx, None)
        valid_region[tuple(slices)] = False
    
    print(f"Valid region voxels (after boundary filtering): {np.sum(valid_region)}")
    
    # Apply the valid region mask to cavity grid
    cavity_grid = cavity_grid & valid_region
    print(f"Remaining cavity voxels (after boundary filtering): {np.sum(cavity_grid)}")
    
    # Use distance transform to find cavity centers
    dist_transform = ndimage.distance_transform_edt(cavity_grid)
    dist_transform_physical = dist_transform * voxel_size
    print(f"Maximum distance to particle: {np.max(dist_transform_physical):.4f} real units")
    
    # Find local maxima in distance transform
    # Use larger neighborhood size to ensure better separation
    neighborhood_size = max(3, int(min_separation / voxel_size))
    local_max = ndimage.maximum_filter(dist_transform_physical, size=neighborhood_size)
    
    # Only consider points that are:
    # 1. Local maxima
    # 2. Above minimum radius threshold
    # 3. At least min_separation away from any particle
    # 4. Within valid region (away from boundaries)
    maxima = (dist_transform_physical == local_max) & \
            (dist_transform_physical >= min_radius) & \
            valid_region
    
    maxima_coords = np.argwhere(maxima)
    print(f"\nFound {len(maxima_coords)} potential cavity centers after initial filtering")
    print(f"  Min radius threshold: {min_radius}")
    print(f"  Neighborhood size for local maxima: {neighborhood_size} voxels")
    
    # 4. Calculate cavity properties
    print("\n4. Calculating cavity properties...")
    cavities = []
    
    # Sort coordinates by distance value (largest first)
    maxima_coords = sorted(
        maxima_coords,
        key=lambda coord: dist_transform_physical[tuple(coord)],
        reverse=True
    )
    
    # Keep track of accepted cavity centers for separation check
    accepted_centers = []
    
    def has_enough_surrounding_particles(coord, radius_voxels):
        """
        Check if a cavity has enough surrounding particles by examining shells around it.
        """
        x, y, z = coord
        inner_r = radius_voxels  # Inner radius is the cavity radius
        outer_r = int(radius_voxels * 2.0)  # Outer radius to check for surrounding particles
        
        # Create a spherical shell mask
        y_grid, x_grid, z_grid = np.ogrid[-outer_r:outer_r+1, -outer_r:outer_r+1, -outer_r:outer_r+1]
        dist_from_center = np.sqrt(x_grid*x_grid + y_grid*y_grid + z_grid*z_grid)
        shell_mask = (dist_from_center > inner_r) & (dist_from_center <= outer_r)
        
        # Get the region to examine
        x_min, x_max = max(0, x-outer_r), min(grid_size, x+outer_r+1)
        y_min, y_max = max(0, y-outer_r), min(grid_size, y+outer_r+1)
        z_min, z_max = max(0, z-outer_r), min(grid_size, z+outer_r+1)
        
        # Adjust shell mask to match the actual region we can examine
        mask_x_min, mask_x_max = outer_r - (x - x_min), outer_r + (x_max - x)
        mask_y_min, mask_y_max = outer_r - (y - y_min), outer_r + (y_max - y)
        mask_z_min, mask_z_max = outer_r - (z - z_min), outer_r + (z_max - z)
        shell_mask = shell_mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max, mask_z_min:mask_z_max]
        
        # Get the particle occupancy in this region
        neighborhood = voxel_grid[x_min:x_max, y_min:y_max, z_min:z_max]
        
        # Calculate statistics for the shell region only
        shell_voxels = np.sum(shell_mask)
        particle_voxels_in_shell = np.sum(neighborhood & shell_mask)
        empty_fraction = 1.0 - (particle_voxels_in_shell / shell_voxels)
        
        # Count particles in different directions
        sectors = []
        sector_size = outer_r - inner_r
        
        # Check each sector
        sector_checks = [
            # (slice_x, slice_y, slice_z, name)
            (slice(sector_size,-sector_size), slice(None,sector_size), slice(sector_size,-sector_size), "+y"),
            (slice(sector_size,-sector_size), slice(-sector_size,None), slice(sector_size,-sector_size), "-y"),
            (slice(None,sector_size), slice(sector_size,-sector_size), slice(sector_size,-sector_size), "+x"),
            (slice(-sector_size,None), slice(sector_size,-sector_size), slice(sector_size,-sector_size), "-x"),
            (slice(sector_size,-sector_size), slice(sector_size,-sector_size), slice(None,sector_size), "+z"),
            (slice(sector_size,-sector_size), slice(sector_size,-sector_size), slice(-sector_size,None), "-z")
        ]
        
        sectors_with_particles = 0
        sector_details = []
        
        for sx, sy, sz, name in sector_checks:
            has_particles = np.any(neighborhood[sx, sy, sz])
            if has_particles:
                sectors_with_particles += 1
            sector_details.append((name, has_particles))
        
        # A cavity should have particles in most directions (at least 4 out of 6)
        enough_surrounding_particles = sectors_with_particles >= 4
        
        # The empty fraction check is now more meaningful as it only considers the shell region
        acceptable_empty_fraction = empty_fraction <= max_empty_neighbors_fraction
        
        return (enough_surrounding_particles and acceptable_empty_fraction), \
               sectors_with_particles, empty_fraction, sector_details

    filtered_by_bounds = 0
    filtered_by_particles = 0
    filtered_by_separation = 0
    
    print("\nDetailed cavity analysis:")
    print("========================")
    
    for coord in maxima_coords:
        # Convert to real coordinates
        center = np.array([centers[d][coord[d]] for d in range(3)])
        radius = dist_transform_physical[tuple(coord)]
        radius_voxels = int(radius / voxel_size)
        
        print(f"\nAnalyzing potential cavity at {center}")
        print(f"Radius: {radius:.4f} units ({radius_voxels} voxels)")
        
        # Double check that the center is within structure bounds
        if not all(structure_min[d] + boundary_margin <= center[d] <= structure_max[d] - boundary_margin for d in range(3)):
            filtered_by_bounds += 1
            print("  Failed: Outside boundary margins")
            continue
        
        # Check if the cavity has enough surrounding particles
        passes_particle_check, sectors_count, empty_fraction, sector_details = has_enough_surrounding_particles(coord, radius_voxels)
        
        print(f"  Shell analysis:")
        print(f"    Empty fraction: {empty_fraction:.3f} (max allowed: {max_empty_neighbors_fraction})")
        print(f"    Sectors with particles: {sectors_count}/6 (minimum needed: 4)")
        print("    Sector details:")
        for name, has_particles in sector_details:
            print(f"      {name}: {'✓' if has_particles else '✗'}")
        
        if not passes_particle_check:
            filtered_by_particles += 1
            if empty_fraction > max_empty_neighbors_fraction:
                print("  Failed: Too much empty space in shell")
            if sectors_count < 4:
                print("  Failed: Not enough surrounding particles")
            continue
        
        # Check separation from existing cavities
        too_close = False
        for existing_center in accepted_centers:
            separation = np.linalg.norm(center - existing_center)
            if separation < min_separation:
                too_close = True
                filtered_by_separation += 1
                print(f"  Failed: Too close to existing cavity (separation: {separation:.4f})")
                break
        
        if too_close:
            continue
            
        print("  Passed all checks!")
        
        # Add cavity and its center
        cavities.append({
            'center': center,
            'radius': radius,
            'volume': (4/3) * np.pi * radius**3
        })
        accepted_centers.append(center)
    
    print(f"\nFiltering statistics:")
    print(f"  Filtered by boundary check: {filtered_by_bounds}")
    print(f"  Filtered by particle check: {filtered_by_particles}")
    print(f"  Filtered by separation check: {filtered_by_separation}")
    print(f"  Remaining cavities: {len(cavities)}")
    
    print("\nFound cavities:")
    for i, cavity in enumerate(cavities):
        print(f"Cavity {i+1}:")
        print(f"  Center: {cavity['center']}")
        print(f"  Radius: {cavity['radius']:.4f}")
        print(f"  Volume: {cavity['volume']:.4f}")
        # Print distance to structure boundaries
        distances_to_bounds = np.minimum(
            cavity['center'] - structure_min,
            structure_max - cavity['center']
        )
        print(f"  Min distance to boundary: {np.min(distances_to_bounds):.4f}")
    
    # Visualize
    if len(cavities) > 0:
        centers = [c['center'] for c in cavities]
        radii = [c['radius'] for c in cavities]
        plot_cavity_spheres(
            particles, centers, radii,
            shape_vertices, shape_color, True, 0.3,
            None, geometry_func, truncation_factor
        )
    
    return cavities

def detect_clathrate_cavities(particles, shape_vertices=None, shape_color=None, 
                             grid_size=64, padding=0.1, cavity_threshold=0.5, 
                             min_cavity_size=10, simulation_data=None,
                             geometry_func=None, truncation_factor=None,
                             keep_largest_cavity_only=False):
    """
    Detect clathrate cavities in a unit cell by finding empty spaces between particles.
    This is a wrapper around detect_simple_cavities for backward compatibility.
    
    Args:
        particles: List of (position, quaternion) tuples
        shape_vertices: numpy array of vertex coordinates from file
        shape_color: color string from file
        grid_size: number of voxels per dimension for cavity detection
        padding: fraction of box size to pad on each side
        cavity_threshold: threshold for considering a region as cavity (0-1)
        min_cavity_size: minimum number of connected voxels to consider as cavity
        simulation_data: Dictionary containing simulation metadata
        geometry_func: function to get geometry
        truncation_factor: truncation parameter
        keep_largest_cavity_only: whether to keep only the largest cavity
    
    Returns:
        cavity_voxels: 3D numpy array with cavity regions marked
        cavity_centers: List of cavity center coordinates
        cavity_volumes: List of cavity volumes
        cavity_radii: List of cavity radii
        voxel_grid: Original particle voxel grid
        edges: Grid edges for reference
    """
    # Convert parameters to detect_simple_cavities format
    min_radius = (min_cavity_size ** (1/3)) * (padding / grid_size)  # Estimate min_radius from min_cavity_size
    min_separation = min_radius * 2  # Reasonable separation based on min_radius
    boundary_margin = padding  # Use padding as boundary margin
    max_empty_neighbors_fraction = 1 - cavity_threshold  # Convert threshold to empty fraction
    
    # Call detect_simple_cavities with converted parameters
    cavities = detect_simple_cavities(
        particles=particles,
        shape_vertices=shape_vertices,
        shape_color=shape_color,
        grid_size=grid_size,
        padding=padding,
        geometry_func=geometry_func,
        truncation_factor=truncation_factor,
        min_radius=min_radius,
        min_separation=min_separation,
        boundary_margin=boundary_margin,
        min_surrounding_particles=2,  # Relaxed requirement for backward compatibility
        max_empty_neighbors_fraction=max_empty_neighbors_fraction,
        debug=False  # Don't show debug output in compatibility mode
    )
    
    # Create a voxel grid representation of the cavities
    positions = np.array([pos for pos, _ in particles])
    min_corner = positions.min(axis=0)
    max_corner = positions.max(axis=0)
    box_size = max_corner - min_corner
    
    # Create grid edges
    min_corner = min_corner - padding * box_size
    max_corner = max_corner + padding * box_size
    edges = [np.linspace(min_corner[d], max_corner[d], grid_size + 1) for d in range(3)]
    centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
    
    # Create empty cavity grid
    cavity_voxels = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    
    # Fill cavity grid based on cavity spheres
    z, y, x = np.meshgrid(
        np.arange(grid_size),
        np.arange(grid_size),
        np.arange(grid_size),
        indexing='ij'
    )
    
    for cavity in cavities:
        center = cavity['center']
        radius = cavity['radius']
        
        # Convert center to grid coordinates
        center_grid = np.array([
            np.searchsorted(centers[d], center[d])
            for d in range(3)
        ])
        
        # Calculate distances to center
        distances = np.sqrt(
            ((x - center_grid[0]) * (edges[0][1] - edges[0][0]))**2 +
            ((y - center_grid[1]) * (edges[1][1] - edges[1][0]))**2 +
            ((z - center_grid[2]) * (edges[2][1] - edges[2][0]))**2
        )
        
        # Mark voxels within radius
        cavity_voxels |= (distances <= radius)
    
    # Create particle grid
    voxel_grid = np.zeros_like(cavity_voxels)
    if geometry_func is not None and truncation_factor is not None:
        vertices, _, _ = geometry_func(shape_vertices, None, truncation_factor)
    else:
        vertices = shape_vertices
    
    for pos, quat in particles:
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        verts = r.apply(vertices) + np.array(pos)
        
        # Get bounding box in grid coordinates
        minv = verts.min(axis=0)
        maxv = verts.max(axis=0)
        idx_min = [max(0, np.searchsorted(centers[d], minv[d], side='left')) for d in range(3)]
        idx_max = [min(grid_size, np.searchsorted(centers[d], maxv[d], side='right')) for d in range(3)]
        
        hull = ConvexHull(verts)
        
        for i in range(idx_min[0], idx_max[0]):
            for j in range(idx_min[1], idx_max[1]):
                for k in range(idx_min[2], idx_max[2]):
                    point = np.array([centers[0][i], centers[1][j], centers[2][k]])
                    inside = True
                    for eq in hull.equations:
                        if np.dot(eq[:3], point) + eq[3] > 1e-10:
                            inside = False
                            break
                    if inside:
                        voxel_grid[i,j,k] = True
    
    # Extract cavity properties
    cavity_centers = [c['center'] for c in cavities]
    cavity_volumes = [c['volume'] for c in cavities]
    cavity_radii = [c['radius'] for c in cavities]
    
    if keep_largest_cavity_only and len(cavities) > 0:
        # Find the largest cavity
        largest_idx = np.argmax(cavity_volumes)
        cavity_centers = [cavity_centers[largest_idx]]
        cavity_volumes = [cavity_volumes[largest_idx]]
        cavity_radii = [cavity_radii[largest_idx]]
        # Update cavity_voxels to only show largest cavity
        cavity_voxels = np.zeros_like(cavity_voxels)
        center = cavity_centers[0]
        radius = cavity_radii[0]
        center_grid = np.array([
            np.searchsorted(centers[d], center[d])
            for d in range(3)
        ])
        distances = np.sqrt(
            ((x - center_grid[0]) * (edges[0][1] - edges[0][0]))**2 +
            ((y - center_grid[1]) * (edges[1][1] - edges[1][0]))**2 +
            ((z - center_grid[2]) * (edges[2][1] - edges[2][0]))**2
        )
        cavity_voxels = (distances <= radius)
    
    return cavity_voxels, cavity_centers, cavity_volumes, cavity_radii, voxel_grid, edges

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
    
    cavity_voxels, cavity_centers, cavity_volumes, cavity_radii, particle_voxels, edges = detect_simple_cavities(
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
    cavity_voxels, cavity_centers, cavity_volumes, cavity_radii, _, edges = detect_simple_cavities(
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


def find_central_cavity(particles, shape_vertices=None, shape_color=None, 
                      grid_size=128, padding=0.15, simulation_data=None, 
                      geometry_func=None, truncation_factor=None):
    """
    Find a cavity at the center of the object.
    
    Args:
        particles: List of (position, quaternion) tuples
        shape_vertices: numpy array of vertex coordinates from file
        shape_color: color string from file
        grid_size: number of voxels per dimension
        padding: fraction of box size to pad on each side
        simulation_data: Dictionary containing simulation metadata
        geometry_func: function to get geometry
        truncation_factor: truncation parameter
    
    Returns:
        cavity_center: [x, y, z] coordinates of cavity center
        cavity_radius: radius of the cavity
        cavity_volume: volume of the cavity
    """
    # First, find the center of the object
    positions = np.array([pos for pos, _ in particles])
    object_center = np.mean(positions, axis=0)
    
    print(f"Object center: {object_center}")
    
    # Create the particle voxel grid
    print("Creating particle voxel grid...")
    voxel_grid, edges = voxelize_particles(particles, grid_size, padding, shape_vertices, None, geometry_func, truncation_factor)
    
    # Invert the grid: particles are 0, empty space is 1
    cavity_grid = 1 - voxel_grid
    
    # Convert object center to grid coordinates
    centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
    center_idx = [
        np.argmin(np.abs(centers[d] - object_center[d])) 
        for d in range(3)
    ]
    
    print(f"Center grid indices: {center_idx}")
    
    # Create a distance mask that prioritizes the center
    z, y, x = np.ogrid[:grid_size, :grid_size, :grid_size]
    dist_from_center = np.sqrt(
        (x - center_idx[0])**2 + 
        (y - center_idx[1])**2 + 
        (z - center_idx[2])**2
    )
    
    # Weight the cavity grid by distance from center
    weighted_cavity = cavity_grid * (1 / (1 + dist_from_center))
    
    # Find the maximum value in the weighted cavity grid
    max_idx = np.unravel_index(np.argmax(weighted_cavity), weighted_cavity.shape)
    
    # Convert to real coordinates
    cavity_center = np.array([
        centers[0][max_idx[0]],
        centers[1][max_idx[1]],
        centers[2][max_idx[2]]
    ])
    
    print(f"Found cavity center at: {cavity_center}")
    
    # Calculate the cavity radius and volume
    sphere_radius, sphere_volume = calculate_cavity_sphere_radius(
        particles, cavity_center, shape_vertices,
        simulation_data=simulation_data,
        geometry_func=geometry_func,
        truncation_factor=truncation_factor
    )
    
    print(f"Cavity radius: {sphere_radius}")
    print(f"Cavity volume: {sphere_volume}")
    
    # Visualize the cavity
    plot_cavity_spheres(
        particles, [cavity_center], [sphere_radius],
        shape_vertices, shape_color, True, 0.3,
        simulation_data, geometry_func, truncation_factor
    )
    
    return cavity_center, sphere_radius, sphere_volume

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
    
    # Example 8: Find central cavity
    print("\n8. Finding central cavity...")
    find_central_cavity(
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

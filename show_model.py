#%%
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
import datetime

# Parse shape definition from file
def parse_shape_definition(shape_line):
    """
    Parse a shape definition line like:
    shape "poly3d 5 0.5 0.288675 0 -0.5 0.288675 0 0 -0.57735 0 0 0 0.204124 0 0 -0.204124 ffff0000"
    
    Returns:
        vertices: numpy array of vertex coordinates
        color: color string in rgba format
    """
    # Remove "shape" prefix and quotes if present
    if shape_line.startswith('shape '):
        shape_line = shape_line[6:]  # Remove "shape " prefix
    if shape_line.startswith('"') and shape_line.endswith('"'):
        shape_line = shape_line[1:-1]  # Remove quotes
    
    parts = shape_line.strip().split()
    
    if len(parts) < 2 or parts[0] != "poly3d":
        raise ValueError("Invalid shape definition format")
    
    num_vertices = int(parts[1])
    
    # Parse vertex coordinates (3 floats per vertex)
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

# Parse data into (position, quaternion) list and extract shape definition
def parse_particles_and_shape(filename):
    """
    Parse particle data and extract shape definition from the same file.
    
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
            
            # Parse date (line 1)
            if line.startswith('//date:'):
                simulation_data['date'] = line[7:].strip()
                continue
            
            # Parse simulation statistics (line 2-3)
            if line.startswith('#[data]'):
                # Store column headers
                simulation_data['data_columns'] = line[7:].strip().split('\t')
                continue
            
            # Parse simulation values (line 3)
            if line_num == 3 and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 23:  # Expected number of columns
                    simulation_data['simulation_values'] = {
                        'steps': int(parts[0]),
                        'time': float(parts[1]),
                        'volume': float(parts[2]),
                        'packing': float(parts[3]),
                        'pressure': float(parts[4]),
                        'msd': float(parts[5]),
                        'delta_x': float(parts[6]),
                        'delta_q': float(parts[7]),
                        'delta_v': float(parts[8]),
                        'accept_x': float(parts[9]),
                        'accept_q': float(parts[10]),
                        'accept_v': float(parts[11]),
                        'ensemble': int(parts[12]),
                        'shear': float(parts[13]),
                        'overlaps': int(parts[14]),
                        'x_length': float(parts[15]),
                        'y_length': float(parts[16]),
                        'z_length': float(parts[17]),
                        'xy_angle': float(parts[18]),
                        'xz_angle': float(parts[19]),
                        'yz_angle': float(parts[20]),
                        'rng_state': parts[21],
                        'rng_state_w': parts[22]
                    }
                continue
            
            # Parse translation (line 6)
            if line.startswith('translation'):
                parts = line.split()
                if len(parts) == 4:
                    simulation_data['translation'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                continue
            
            # Parse zoom factor (line 7)
            if line.startswith('zoomFactor'):
                parts = line.split()
                if len(parts) == 2:
                    simulation_data['zoom_factor'] = float(parts[1])
                continue
            
            # Parse box matrix (line 8)
            if line.startswith('boxMatrix'):
                parts = line.split()
                if len(parts) == 10:  # 9 matrix elements + 'boxMatrix'
                    matrix_values = [float(x) for x in parts[1:]]
                    simulation_data['box_matrix'] = np.array(matrix_values).reshape(3, 3)
                continue
            
            # Check for shape definition (line 9)
            if line.startswith('shape'):
                # Extract everything after the first whitespace following 'shape'
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
            
            # Parse particle data (lines 10+)
            parts = line.split()
            if len(parts) < 7:
                continue
            
            try:
                # Only treat as particle if first 7 columns are floats
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
            print(f"  Box Angles: {sim_vals['xy_angle']:.1f}°, {sim_vals['xz_angle']:.1f}°, {sim_vals['yz_angle']:.1f}°")
        if 'translation' in simulation_data:
            print(f"  Translation: {simulation_data['translation']}")
        if 'zoom_factor' in simulation_data:
            print(f"  Zoom Factor: {simulation_data['zoom_factor']}")
        if 'box_matrix' in simulation_data:
            print(f"  Box Matrix:\n{simulation_data['box_matrix']}")
    
    return particles, shape_vertices, shape_color, simulation_data

# Define triangular bipyramid shape from file or use default
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
            # Vertices: [V0=base1, V1=base2, V2=base3, V3=top_apex, V4=bottom_apex]
            # Based on the shape file: V0,V1,V2 form triangular base, V3=top, V4=bottom
            faces = [
                [3, 0, 1],  # Top apex to base edge 0-1
                [3, 1, 2],  # Top apex to base edge 1-2
                [3, 2, 0],  # Top apex to base edge 2-0
                [4, 0, 1],  # Bottom apex to base edge 0-1
                [4, 1, 2],  # Bottom apex to base edge 1-2
                [4, 2, 0]   # Bottom apex to base edge 2-0
            ]
        else:
            # For other shapes, create a simple triangulation
            faces = []
            
        return shape_vertices, faces, shape_color or 'rgba(150,150,250,0.6)'
    
    # Default triangular bipyramid geometry
    top = [0, 0, 1]      # V0: Top apex
    bottom = [0, 0, -1]  # V4: Bottom apex
    base = [
        [np.cos(a), np.sin(a), 0]
        for a in np.linspace(0, 2 * np.pi, 4)[:-1]  # V1, V2, V3: Base triangle
    ]
    vertices = np.array([top] + base + [bottom]) * 0.2
    
    # Correct face definitions for triangular bipyramid
    faces = [
        [0, 1, 2],  # Top apex to base edge 1-2
        [0, 2, 3],  # Top apex to base edge 2-3
        [0, 3, 1],  # Top apex to base edge 3-1
        [4, 1, 2],  # Bottom apex to base edge 1-2
        [4, 2, 3],  # Bottom apex to base edge 2-3
        [4, 3, 1]   # Bottom apex to base edge 3-1
    ]
    color = 'rgba(150,150,250,0.6)'
    return vertices, faces, color

# Calculate unit cell dimensions from particle positions or simulation metadata
def get_unit_cell_dimensions(particles, simulation_data=None):
    """
    Calculate unit cell dimensions from simulation metadata or particle positions.
    
    Args:
        particles: List of (position, quaternion) tuples
        simulation_data: Dictionary containing simulation metadata
    
    Returns:
        cell_dims: [x, y, z] dimensions of the unit cell
    """
    # If simulation metadata is available, use the box dimensions from there
    if simulation_data and 'simulation_values' in simulation_data:
        sim_vals = simulation_data['simulation_values']
        return [sim_vals['x_length'], sim_vals['y_length'], sim_vals['z_length']]
    
    # Fall back to calculating from particle positions
    if not particles:
        return [1.0, 1.0, 1.0]  # Default dimensions
    
    positions = np.array([pos for pos, _ in particles])
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    
    # Add some padding to ensure complete unit cell
    cell_dims = max_coords - min_coords
    return cell_dims

# Duplicate unit cell in multiple directions
def duplicate_unit_cell(particles, nx=1, ny=1, nz=1, simulation_data=None):
    """
    Duplicate the unit cell nx, ny, nz times in x, y, z directions.
    
    Args:
        particles: List of (position, quaternion) tuples
        nx: Number of repetitions in x direction
        ny: Number of repetitions in y direction  
        nz: Number of repetitions in z direction
        simulation_data: Dictionary containing simulation metadata
    
    Returns:
        List of duplicated particles with cell indices
    """
    if not particles:
        return []
    
    # Calculate unit cell dimensions
    cell_dims = get_unit_cell_dimensions(particles, simulation_data)
    
    duplicated_particles = []
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                # Calculate offset for this cell
                offset = np.array([ix * cell_dims[0], iy * cell_dims[1], iz * cell_dims[2]])
                
                # Duplicate particles with offset
                for pos, quat in particles:
                    new_pos = tuple(np.array(pos) + offset)
                    duplicated_particles.append((new_pos, quat, (ix, iy, iz)))
    
    return duplicated_particles

# Generate mesh trace for a single particle
def get_mesh_trace(pos, quat, vertices, faces, color='rgba(150,150,250,0.6)'):
    if not faces:
        print("Warning: No faces provided for mesh trace. Skipping this particle.")
        return go.Scatter3d()
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # (x, y, z, w)
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

# Generate color based on cell position
def get_cell_color(ix, iy, iz, nx, ny, nz):
    """Generate a color based on the cell position for visualization"""
    if nx == 1 and ny == 1 and nz == 1:
        return 'rgba(150,150,250,0.6)'  # Single cell - default color
    
    # Create a color gradient based on position
    r = int(100 + 155 * ix / max(1, nx-1))
    g = int(100 + 155 * iy / max(1, ny-1))
    b = int(100 + 155 * iz / max(1, nz-1))
    
    return f'rgba({r},{g},{b},0.6)'

# Main plotting function
def plot_particles(particles, nx=1, ny=1, nz=1, show_duplicated=True, shape_vertices=None, shape_color=None, color_by_cell=False, simulation_data=None):
    """
    Plot particles with optional unit cell duplication.
    
    Args:
        particles: List of (position, quaternion) tuples
        nx: Number of repetitions in x direction
        ny: Number of repetitions in y direction
        nz: Number of repetitions in z direction
        show_duplicated: Whether to show duplicated cells or just the original
        shape_vertices: numpy array of vertex coordinates from file
        shape_color: color string from file
        color_by_cell: If True, color each cell differently; if False, use shape_color for all
        simulation_data: Dictionary containing simulation metadata
    """
    vertices, faces, default_color = get_bipyramid_geometry(shape_vertices, shape_color)
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

    # Update title based on duplication
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

def point_in_polyhedron(point, vertices, faces):
    """
    Check if a point is inside a convex polyhedron defined by vertices and faces.
    Args:
        point: (3,) array
        vertices: (N, 3) array
        faces: list of lists of vertex indices
    Returns:
        True if inside, False otherwise
    """
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    # For each face, check if point is on the inner side
    for eq in hull.equations:
        if np.dot(eq[:3], point) + eq[3] > 1e-8:
            return False
    return True

def voxelize_particles(particles, grid_size=64, padding=0.1, shape_vertices=None, shape_faces=None):
    """
    Convert particle positions to a 3D voxel grid with solid bipyramids (density 1 inside, 0 outside), vectorized for speed.
    Args:
        particles: list of (position, quaternion) tuples
        grid_size: number of voxels per dimension
        padding: fraction of box size to pad on each side
        shape_vertices: (N, 3) array of bipyramid vertices (unit shape)
        shape_faces: list of face indices
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
    
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial import ConvexHull
    # For each particle, fill in the voxels inside the transformed bipyramid
    for pos, quat in particles:
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # (x, y, z, w)
        verts = r.apply(shape_vertices) + np.array(pos)
        hull = ConvexHull(verts)
        eqs = hull.equations  # shape (nfaces, 4)
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
        pts = np.stack([XX, YY, ZZ], axis=-1).reshape(-1, 3)  # (Npts, 3)
        # Check all points at once: (Nfaces, 3) @ (Npts, 3).T + (Nfaces, 1)
        inside = np.all((eqs[:,:3] @ pts.T + eqs[:,3:]) <= 1e-8, axis=0)
        # Set all inside voxels to 1
        idxs = np.argwhere(inside)
        for flat_idx in idxs:
            # Convert flat index to 3D index
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
    
    # Get frequency arrays
    nx, ny = projection.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    
    return fft_mag, kx, ky

def create_projection_heatmap(projection, title, projection_type='sum', colorbar_x=1.1, colorbar_y=0.5):
    """
    Create a heatmap trace for a 2D projection.
    Args:
        projection: 2D numpy array
        title: title for the plot
        projection_type: type of projection for colorbar label
        colorbar_x: x position of colorbar
        colorbar_y: y position of colorbar
    Returns:
        go.Heatmap trace
    """
    return go.Heatmap(
        z=projection,
        colorscale='Viridis',
        showscale=True,
        # colorbar=dict(
        #     title=f'{projection_type.capitalize()} Projection',
        #     x=colorbar_x,
        #     y=colorbar_y,
        #     len=0.8,
        #     thickness=20
        # )
    )

def create_fft_heatmap(projection, log_scale=True, threshold=0.1, colorbar_x=1.1, colorbar_y=0.5):
    """
    Create a heatmap trace for a 2D FFT.
    Args:
        projection: 2D numpy array
        log_scale: whether to plot log10(magnitude)
        threshold: fraction of max to threshold for visualization
        colorbar_x: x position of colorbar
        colorbar_y: y position of colorbar
    Returns:
        go.Heatmap trace
    """
    fft_mag, kx, ky = compute_2d_fft(projection)
    
    if log_scale:
        fft_mag = np.log10(fft_mag + 1e-6)
    
    # Threshold for visualization
    vmax = fft_mag.max()
    mask = fft_mag > (threshold * vmax)
    fft_mag[~mask] = 0
    
    return go.Heatmap(
        z=fft_mag,
        x=kx,
        y=ky,
        colorscale='Viridis',
        showscale=True,
        # colorbar=dict(
        #     title='FFT Magnitude (log scale)' if log_scale else 'FFT Magnitude',
        #     x=colorbar_x,
        #     y=colorbar_y,
        #     len=0.8,
        #     thickness=20
        # )
    )

def plot_projections_with_ffts(projections, log_scale=True, threshold=0.1):
    """
    Plot original 2D projections alongside their corresponding FFTs.
    Args:
        projections: dict with 'xy', 'xz', 'yz' keys
        log_scale: whether to plot log10(magnitude) for FFTs
        threshold: fraction of max to threshold for FFT visualization
    """
    from plotly.subplots import make_subplots
    
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
        # Plot original projection (top row) using the helper function
        projection_trace = create_projection_heatmap(
            proj, f'{plane.upper()} Projection', 'sum', 
            colorbar_x=0.15 + i*0.25, colorbar_y=0.5
        )
        fig.add_trace(projection_trace, row=1, col=i+1)
        
        # Plot FFT (bottom row) using the helper function
        fft_trace = create_fft_heatmap(
            proj, log_scale, threshold,
            colorbar_x=0.15 + i*0.25, colorbar_y=0.0
        )
        fig.add_trace(fft_trace, row=2, col=i+1)
    
    fig.update_layout(
        title='2D Projections and Their FFTs',
        height=800,
        width=1200,
        showlegend=False
    )
    fig.show()

def plot_individual_projection(projection, title, projection_type='sum'):
    """
    Plot a single 2D projection with its own colorbar.
    Args:
        projection: 2D numpy array
        title: title for the plot
        projection_type: type of projection for colorbar label
    """
    fig = go.Figure(data=create_projection_heatmap(projection, title, projection_type))
    
    fig.update_layout(
        title=title,
        xaxis_title='X',
        yaxis_title='Y',
        margin=dict(l=0, r=100, b=0, t=40),  # Increased right margin for colorbar
        height=500,
        width=600
    )
    fig.show()

def plot_individual_fft(projection, title, log_scale=True, threshold=0.1):
    """
    Plot a single 2D FFT with its own colorbar.
    Args:
        projection: 2D numpy array
        title: title for the plot
        log_scale: whether to plot log10(magnitude)
        threshold: fraction of max to threshold for visualization
    """
    fig = go.Figure(data=create_fft_heatmap(projection, log_scale, threshold))
    
    fig.update_layout(
        title=title,
        xaxis_title='kx',
        yaxis_title='ky',
        margin=dict(l=0, r=100, b=0, t=40),  # Increased right margin for colorbar
        height=500,
        width=600
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
        margin=dict(l=0, r=100, b=0, t=40),  # Increased right margin for colorbar
        showlegend=False
    )
    fig.show()

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
        from plotly.subplots import make_subplots
        
        # Create subplots for XY slices
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

def plot_individual_particle(pos, quat, shape_vertices=None, shape_color=None, plot_type='surface', show_axes=True):
    """
    Plot a single particle to visualize its geometry.
    
    Args:
        pos: (x, y, z) position tuple
        quat: (qw, qx, qy, qz) quaternion tuple
        shape_vertices: numpy array of vertex coordinates
        shape_color: color string
        plot_type: 'surface', 'wireframe', or 'both'
        show_axes: whether to show coordinate axes
    """
    vertices, faces, default_color = get_bipyramid_geometry(shape_vertices, shape_color)
    
    # Apply rotation and translation
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # (x, y, z, w)
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
            for i in range(len(face)):
                start = face[i]
                end = face[(i + 1) % len(face)]
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
        marker=dict(size=8, color='red'),
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
        title=f'Single Particle at Position {pos}',
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

def plot_multiple_particles(particles, shape_vertices=None, shape_color=None, max_particles=5, plot_type='surface'):
    """
    Plot multiple individual particles to compare their geometries.
    
    Args:
        particles: list of (position, quaternion) tuples
        shape_vertices: numpy array of vertex coordinates
        shape_color: color string
        max_particles: maximum number of particles to plot
        plot_type: 'surface', 'wireframe', or 'both'
    """
    # Limit number of particles to avoid cluttered plot
    particles_to_plot = particles[:max_particles]
    
    fig = go.Figure()
    
    for particle_idx, (pos, quat) in enumerate(particles_to_plot):
        vertices, faces, default_color = get_bipyramid_geometry(shape_vertices, shape_color)
        
        # Apply rotation and translation
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # (x, y, z, w)
        verts_rot = r.apply(vertices) + np.array(pos)
        
        if plot_type in ['surface', 'both']:
            # Create surface mesh
            x, y, z = verts_rot[:, 0], verts_rot[:, 1], verts_rot[:, 2]
            i, j, k = zip(*faces)
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=0.7,
                color=default_color,
                flatshading=True,
                showscale=False,
                name=f'Particle {particle_idx+1}'
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
                    line=dict(color='black', width=2),
                    showlegend=False
                ))
        
        # Add particle center as a point
        fig.add_trace(go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[pos[2]],
            mode='markers',
            marker=dict(size=5, color='red'),
            text=[f'P{particle_idx+1}'],
            showlegend=False
        ))
    
    fig.update_layout(
        title=f'Multiple Particles ({len(particles_to_plot)} shown)',
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

def plot_particle_geometry_analysis(particles, shape_vertices=None, shape_color=None, num_samples=3):
    """
    Analyze and plot individual particle geometries with detailed information.
    
    Args:
        particles: list of (position, quaternion) tuples
        shape_vertices: numpy array of vertex coordinates
        shape_color: color string
        num_samples: number of particles to analyze
    """
    # Sample particles for analysis
    sample_particles = particles[:num_samples]
    
    for i, (pos, quat) in enumerate(sample_particles):
        print(f"\n{'='*50}")
        print(f"PARTICLE {i+1} ANALYSIS")
        print(f"{'='*50}")
        print(f"Position: {pos}")
        print(f"Quaternion (w,x,y,z): {quat}")
        
        # Calculate rotation matrix
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rotation_matrix = r.as_matrix()
        print(f"Rotation Matrix:\n{rotation_matrix}")
        
        # Get Euler angles
        euler_angles = r.as_euler('xyz', degrees=True)
        print(f"Euler Angles (xyz, degrees): {euler_angles}")
        
        # Plot the individual particle
        plot_individual_particle(pos, quat, shape_vertices, shape_color, plot_type='both', show_axes=True)

def create_truncated_bipyramid(original_vertices, truncation_factor=0.3):
    """
    Create a truncated triangular bipyramid using the mathematically correct method.
    
    Based on the paper's approach:
    - Uses convex hull geometry
    - Preserves threefold rotational symmetry
    - Creates 18 vertices by interpolating along edges
    - Parameter t: 0 = no truncation, 1 = truncate to edge midpoints
    
    Args:
        original_vertices: numpy array of original vertex coordinates (5 vertices)
        truncation_factor: truncation parameter t (0-1)
    
    Returns:
        vertices: numpy array of new vertex coordinates (18 vertices)
        faces: list of face indices (convex hull faces)
    """
    if len(original_vertices) != 5:
        raise ValueError("Original vertices must have exactly 5 vertices")
    
    # Original vertex arrangement: [V0=base1, V1=base2, V2=base3, V3=top_apex, V4=bottom_apex]
    # Map to paper notation: o=V3(top), p=V0, q=V1, r=V2, s=V4(bottom)
    o = original_vertices[3]  # Top apex
    p = original_vertices[0]  # Base vertex 1
    q = original_vertices[1]  # Base vertex 2
    r = original_vertices[2]  # Base vertex 3
    s = original_vertices[4]  # Bottom apex
    
    t = truncation_factor
    
    # Create 18 new vertices by interpolating along edges
    # Each edge gets two new vertices, positioned symmetrically
    new_vertices = []
    
    # Edge o-p (top apex to base vertex 1)
    op = (1 - t/2) * o + (t/2) * p  # Closer to o
    po = (1 - t/2) * p + (t/2) * o  # Closer to p
    
    # Edge o-q (top apex to base vertex 2)
    oq = (1 - t/2) * o + (t/2) * q  # Closer to o
    qo = (1 - t/2) * q + (t/2) * o  # Closer to q
    
    # Edge o-r (top apex to base vertex 3)
    or_vertex = (1 - t/2) * o + (t/2) * r  # Closer to o
    ro = (1 - t/2) * r + (t/2) * o  # Closer to r
    
    # Edge s-p (bottom apex to base vertex 1)
    sp = (1 - t/2) * s + (t/2) * p  # Closer to s
    ps = (1 - t/2) * p + (t/2) * s  # Closer to p
    
    # Edge s-q (bottom apex to base vertex 2)
    sq = (1 - t/2) * s + (t/2) * q  # Closer to s
    qs = (1 - t/2) * q + (t/2) * s  # Closer to q
    
    # Edge s-r (bottom apex to base vertex 3)
    sr = (1 - t/2) * s + (t/2) * r  # Closer to s
    rs = (1 - t/2) * r + (t/2) * s  # Closer to r
    
    # Edge q-r (base vertex 2 to base vertex 3)
    qr = (1 - t/2) * q + (t/2) * r  # Closer to q
    rq = (1 - t/2) * r + (t/2) * q  # Closer to r
    
    # Edge r-p (base vertex 3 to base vertex 1)
    rp = (1 - t/2) * r + (t/2) * p  # Closer to r
    pr = (1 - t/2) * p + (t/2) * r  # Closer to p
    
    # Edge p-q (base vertex 1 to base vertex 2)
    pq = (1 - t/2) * p + (t/2) * q  # Closer to p
    qp = (1 - t/2) * q + (t/2) * p  # Closer to q
    
    # Collect all vertices
    new_vertices = [
        op, po,  # Edge o-p
        oq, qo,  # Edge o-q
        or_vertex, ro,  # Edge o-r
        sp, ps,  # Edge s-p
        sq, qs,  # Edge s-q
        sr, rs,  # Edge s-r
        qr, rq,  # Edge q-r
        rp, pr,  # Edge r-p
        pq, qp   # Edge p-q
    ]
    
    new_vertices = np.array(new_vertices)
    
    # Use convex hull to find faces
    from scipy.spatial import ConvexHull
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
        # Create truncated version of the shape from file
        vertices, faces = create_truncated_bipyramid(shape_vertices, truncation_factor)
        return vertices, faces, shape_color or 'rgba(150,150,250,0.6)'
    
    # Create default truncated bipyramid
    top = [0, 0, 1]      # V3: Top apex
    bottom = [0, 0, -1]  # V4: Bottom apex
    base = [
        [np.cos(a), np.sin(a), 0]
        for a in np.linspace(0, 2 * np.pi, 4)[:-1]  # V0, V1, V2: Base triangle
    ]
    default_vertices = np.array([base[0], base[1], base[2], top, bottom]) * 0.2
    
    vertices, faces = create_truncated_bipyramid(default_vertices, truncation_factor)
    color = 'rgba(150,150,250,0.6)'
    return vertices, faces, color

def get_truncated_bipyramid_volume_surface(truncation_factor):
    """
    Calculate volume and surface area of truncated triangular bipyramid.
    
    Based on the paper's analytical formulas:
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
    from plotly.subplots import make_subplots
    
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

def plot_truncated_particle(pos, quat, shape_vertices=None, shape_color=None, truncation_factor=0.3, plot_type='surface', show_axes=True):
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
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # (x, y, z, w)
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
        nx: Number of repetitions in x direction
        ny: Number of repetitions in y direction
        nz: Number of repetitions in z direction
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


filename = 'C:\\Users\\b304014\\Software\\blee\\models\\ClaS_bipyramid_averaged.pos'  # <-- Replace with actual file path

# Parse both particles and shape definition from the same file
particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(filename)
# Get faces for the shape
_, shape_faces, _ = get_bipyramid_geometry(shape_vertices, shape_color)

# Example usage - you can modify these parameters
print("Plotting single unit cell...")
plot_particles(particles, nx=3, ny=3, nz=3, show_duplicated=False, 
                shape_vertices=shape_vertices, shape_color=shape_color, simulation_data=simulation_data)

#%%
# Example: Individual Particle Geometry Analysis
print("\n" + "="*50)
print("INDIVIDUAL PARTICLE GEOMETRY ANALYSIS")
print("="*50)

# Plot individual particles to see triangular bipyramid geometry
print("Plotting individual particles...")

# Plot first particle with detailed geometry
print("Plotting first particle with wireframe and surface...")
plot_individual_particle(
    pos=particles[0][0],  # First particle position
    quat=particles[0][1],  # First particle quaternion
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    plot_type='both',  # Show both surface and wireframe
    show_axes=True
)

# Plot multiple particles for comparison
print("Plotting multiple particles for comparison...")
plot_multiple_particles(
    particles=particles[:5],  # First 5 particles
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    max_particles=5,
    plot_type='both'
)

# Detailed analysis of individual particles
print("Performing detailed geometry analysis...")
plot_particle_geometry_analysis(
    particles=particles,
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    num_samples=3  # Analyze first 3 particles
)
#%%
# print("Plotting 2x2x2 duplicated unit cells...")
# plot_particles(particles, nx=2, ny=2, nz=2, show_duplicated=True, 
#                 shape_vertices=shape_vertices, shape_color=shape_color)

# To color by cell, use: color_by_cell=True
# plot_particles(particles, nx=2, ny=2, nz=2, show_duplicated=True, 
#                shape_vertices=shape_vertices, shape_color=shape_color, color_by_cell=True)

# print("Plotting 3x1x2 duplicated unit cells...")
# plot_particles(particles, nx=3, ny=1, nz=2, show_duplicated=True, 
#               shape_vertices=shape_vertices, shape_color=shape_color)

# print("Plotting 3x3x3 duplicated unit cells...")
# plot_particles(particles, nx=3, ny=3, nz=3, show_duplicated=True, 
#               shape_vertices=shape_vertices, shape_color=shape_color)

# 3D FFT of the assembly
print("Computing 3D FFT of the assembly...")
# You can choose which assembly to FFT (e.g., single cell, duplicated, etc.)
all_particles = duplicate_unit_cell(particles, nx=1, ny=1, nz=1, simulation_data=simulation_data)
all_positions = [(pos, quat) for pos, quat, _ in all_particles]
voxel_grid, edges = voxelize_particles(all_positions, grid_size=128, shape_vertices=shape_vertices, shape_faces=shape_faces)
plot_fft_magnitude(voxel_grid, edges, log_scale=True, threshold=0.65)

# 2D Projection FFTs
print("Computing 2D projection FFTs...")
projections = create_2d_projections(voxel_grid, projection_type='sum')
print("Plotting projections with their corresponding FFTs...")
plot_projections_with_ffts(projections, log_scale=True, threshold=0.1)

# # Example of plotting individual projections with separate colorbars
# print("\nPlotting individual projections with separate colorbars...")
# plot_individual_projection(projections['xy'], 'XY Projection (Sum)', 'sum')
# plot_individual_projection(projections['xz'], 'XZ Projection (Sum)', 'sum')
# plot_individual_projection(projections['yz'], 'YZ Projection (Sum)', 'sum')

# # Example of plotting individual FFTs with separate colorbars
# print("\nPlotting individual FFTs with separate colorbars...")
# plot_individual_fft(projections['xy'], 'XY Projection FFT', log_scale=True, threshold=0.1)
# plot_individual_fft(projections['xz'], 'XZ Projection FFT', log_scale=True, threshold=0.1)
# plot_individual_fft(projections['yz'], 'YZ Projection FFT', log_scale=True, threshold=0.1)

# %%
import matplotlib.pyplot as plt
import matplotlib.colors as colors
fig,ax = plt.subplots(1,3,figsize=(15,5))
# Create vignette effect
y, x = np.ogrid[:projections['xy'].shape[0], :projections['xy'].shape[1]]
center_y, center_x = projections['xy'].shape[0]/2, projections['xy'].shape[1]/2
dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
max_dist = np.sqrt(center_x**2 + center_y**2)
vignette = 1 - dist_from_center/max_dist
vignette = np.clip(vignette, 0, 1)

# Apply vignette to projection
vignetted = projections['xy'] * vignette
ax[0].imshow(vignetted)
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(vignetted))**2),norm=colors.LogNorm())
ax[2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(projections['xy']))**2),norm=colors.LogNorm())
plt.show()

# %%
# Example: Save 3D tomogram as TIFF
print("\n" + "="*50)
print("SAVING 3D TOMOGRAM AS TIFF")
print("="*50)

# Create tomogram from the voxelized particles
tomogram_filename = create_tomogram_from_particles(
    particles=all_positions,
    grid_size=128,
    padding=0.1,
    shape_vertices=shape_vertices,
    shape_faces=shape_faces,
    pixel_size=1.0,  # 1 nm per pixel
    filename='bipyramid_tomogram.tif'
)

# Example: Load and verify the tomogram
print("\n" + "="*50)
print("LOADING AND VERIFYING TOMOGRAM")
print("="*50)

loaded_voxel_grid = load_tomogram_tiff(tomogram_filename)

# Verify the loaded data matches the original
print(f"Original voxel grid shape: {voxel_grid.shape}")
print(f"Loaded voxel grid shape: {loaded_voxel_grid.shape}")
print(f"Data matches: {np.array_equal(voxel_grid, loaded_voxel_grid)}")

# Example: Create projections from loaded tomogram
print("\nCreating projections from loaded tomogram...")
loaded_projections = create_2d_projections(loaded_voxel_grid, projection_type='sum')

# Compare original vs loaded projections
print("Comparing original vs loaded projections...")
for plane in ['xy', 'xz', 'yz']:
    original_proj = projections[plane]
    loaded_proj = loaded_projections[plane]
    correlation = np.corrcoef(original_proj.flatten(), loaded_proj.flatten())[0,1]
    print(f"{plane.upper()} projection correlation: {correlation:.6f}")




# %%
# Example: 3D Visualization of the Tomogram
print("\n" + "="*50)
print("3D VISUALIZATION OF TOMOGRAM")
print("="*50)

# Plot 3D isosurface of the tomogram
print("Creating 3D isosurface plot...")
plot_3d_tomogram(tomogram_filename, plot_type='isosurface', threshold=0.3)

# Plot 3D volume rendering of the tomogram
print("Creating 3D volume rendering...")
plot_3d_tomogram(tomogram_filename, plot_type='volume', opacity=0.5)

# Plot 3D orthogonal slices
print("Creating 3D orthogonal slices...")
plot_3d_tomogram(tomogram_filename, plot_type='slices')

# Plot 2D slices
print("Creating 2D slice plots...")
plot_tomogram_slices_2d(tomogram_filename, slice_type='middle')

# Plot all slices in a grid
print("Creating grid of all slices...")
plot_tomogram_slices_2d(tomogram_filename, slice_type='all')





























# %%

# Example: Truncated Particle Analysis
print("\n" + "="*50)
print("TRUNCATED PARTICLE ANALYSIS")
print("="*50)

# Analyze truncation effects using analytical formulas
print("Analyzing truncation effects on volume and surface area...")
analyze_truncation_effects(truncation_factors=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Plot comparison of different truncation levels
print("\nPlotting comparison of different truncation levels...")
plot_truncation_comparison(
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    truncation_factors=[0.0, 0.3, 0.6, 1.0]
)

# Test the truncation method with different parameters
print("\nTesting truncation method...")
for t in [0.1, 0.5, 0.9]:
    print(f"\nTruncation parameter t = {t}:")
    
    # Get truncated geometry
    vertices, faces, color = get_truncated_bipyramid_geometry(shape_vertices, shape_color, t)
    print(f"  Number of vertices: {len(vertices)}")
    print(f"  Number of faces: {len(faces)}")
    
    # Calculate analytical volume and surface area
    volume, surface_area = get_truncated_bipyramid_volume_surface(t)
    print(f"  Volume: {volume:.3f}")
    print(f"  Surface area: {surface_area:.3f}")
    
    # Plot individual truncated particle
    plot_truncated_particle(
        pos=particles[0][0],
        quat=particles[0][1],
        shape_vertices=shape_vertices,
        shape_color=shape_color,
        truncation_factor=t,
        plot_type='both',
        show_axes=True
    )

#%%
# Compare assembly with different truncation levels
print("\nComparing assemblies with different truncation levels...")
for t in [0.2, 0.4, 0.6]:
    print(f"Plotting assembly with truncation t = {t}...")
    plot_particles_with_truncation(
        particles=particles,
        shape_vertices=shape_vertices,
        shape_color=shape_color,
        truncation_factor=t,
        nx=1, ny=1, nz=1,
        show_duplicated=True,
        simulation_data=simulation_data
    )

# Create tomogram with truncation
print("\n" + "="*50)
print("CREATING TOMOGRAM WITH TRUNCATION")
print("="*50)

# Use moderate truncation for tomogram
t_tomogram = 1.0
print(f"Creating tomogram with truncation t = {t_tomogram}...")

# Get truncated geometry
truncated_vertices, truncated_faces, _ = get_truncated_bipyramid_geometry(
    shape_vertices, shape_color, t_tomogram
)

# Create assembly and voxelize
all_particles_truncated = duplicate_unit_cell(particles, nx=1, ny=1, nz=1, simulation_data=simulation_data)
all_positions_truncated = [(pos, quat) for pos, quat, _ in all_particles_truncated]

print("Voxelizing truncated particles...")
voxel_grid_truncated, edges_truncated = voxelize_particles(
    all_positions_truncated, 
    grid_size=128, 
    shape_vertices=truncated_vertices, 
    shape_faces=truncated_faces
)

#%%
# 2D Projection FFTs
print("Computing 2D projection FFTs...")
projections_truncated = create_2d_projections(voxel_grid_truncated, projection_type='sum')
print("Plotting projections with their corresponding FFTs...")
plot_projections_with_ffts(projections_truncated, log_scale=True, threshold=0.1)

# 2D Projection FFTs
print("Computing 2D projection FFTs...")
projections = create_2d_projections(voxel_grid, projection_type='sum')
print("Plotting projections with their corresponding FFTs...")
plot_projections_with_ffts(projections, log_scale=True, threshold=0.1)


# Plot FFT of truncated assembly
print("Computing 3D FFT of truncated assembly...")
plot_fft_magnitude(voxel_grid_truncated, edges_truncated, log_scale=True, threshold=0.7)

# Plot FFT of truncated assembly
print("Computing 3D FFT of truncated assembly...")
plot_fft_magnitude(voxel_grid, edges, log_scale=True, threshold=0.7)



# Check if voxel grids are equal at all points
comparison = voxel_grid_truncated == voxel_grid
are_equal = np.all(comparison)
print(f"Voxel grids are {'equal' if are_equal else 'different'} at all points")
if not are_equal:
    num_diff = np.sum(~comparison)
    total_points = voxel_grid.size
    print(f"Number of different points: {num_diff} out of {total_points} ({(num_diff/total_points)*100:.2f}%)")







#%%
# Create and save truncated tomogram
print("Creating truncated tomogram...")
truncated_tomogram_filename = create_tomogram_from_particles(
    particles=all_positions_truncated,
    grid_size=128,
    padding=0.1,
    shape_vertices=truncated_vertices,
    shape_faces=truncated_faces,
    pixel_size=1.0,
    filename=f'truncated_t{t_tomogram}_bipyramid_tomogram.tif'
)

# Visualize truncated tomogram
print("Visualizing truncated tomogram...")
plot_3d_tomogram(truncated_tomogram_filename, plot_type='isosurface', threshold=0.3)

print("\nTruncation implementation complete!")
print("Key features:")
print("- Uses convex hull geometry for proper face generation")
print("- Preserves threefold rotational symmetry")
print("- Provides analytical volume and surface area formulas")
print("- Creates exactly 18 vertices as specified in the paper")
print("- Parameter t has clear geometric meaning (0=no truncation, 1=edge midpoints)")
# %%
import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    # Read data with space separator and custom column names
    data = pd.read_csv(
        f'C:\\Users\\b304014\\Software\\blee\\models\\SAXS\\truncated_t{t}_bipyramid_tomogram_spherical_avg.dat',
        delim_whitespace=True,
        skiprows=1,
        names=['q', 'Calculated']
    )
    
    plt.plot(data['q'], data['Calculated'], label=f't={t}')

plt.xlabel('q')
plt.xlim(0,3.0)
plt.ylabel('Calculated')
plt.title('SAXS Profiles for Different Truncation Values')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()













# %%

def detect_clathrate_cavities(particles, shape_vertices=None, shape_color=None, 
                             grid_size=64, padding=0.1, cavity_threshold=0.5, 
                             min_cavity_size=10, simulation_data=None):
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
    
    Returns:
        cavity_voxels: 3D numpy array with cavity regions marked
        cavity_centers: List of cavity center coordinates
        cavity_volumes: List of cavity volumes
        voxel_grid: Original particle voxel grid
        edges: Grid edges for reference
    """
    from scipy import ndimage
    from scipy.spatial import ConvexHull
    
    # First, create the particle voxel grid
    print("Creating particle voxel grid...")
    voxel_grid, edges = voxelize_particles(particles, grid_size, padding, shape_vertices, None)
    
    # Invert the grid: particles are 0, empty space is 1
    cavity_grid = 1 - voxel_grid
    
    # Apply morphological operations to clean up the cavity grid
    print("Cleaning up cavity grid...")
    # Remove small noise
    cavity_grid = ndimage.binary_opening(cavity_grid, structure=np.ones((4,4,4)))
    # Fill small holes
    cavity_grid = ndimage.binary_closing(cavity_grid, structure=np.ones((4,4,4)))
    
    # Find connected components (individual cavities)
    print("Finding connected cavity regions...")
    labeled_cavities, num_cavities = ndimage.label(cavity_grid)

    # Remove the largest (outside) cavity
    if num_cavities > 0:
        sizes = ndimage.sum(cavity_grid, labeled_cavities, range(1, num_cavities + 1))
        largest_label = np.argmax(sizes) + 1  # labels start at 1
        labeled_cavities[labeled_cavities == largest_label] = 0
        # Re-label to remove the gap in labels
        labeled_cavities, num_cavities = ndimage.label(labeled_cavities > 0)

    print(f"Found {num_cavities} potential cavity regions (excluding outside)")
    
    cavity_centers = []
    cavity_volumes = []
    cavity_voxels = np.zeros_like(cavity_grid)
    
    for cavity_id in range(1, num_cavities + 1):
        cavity_mask = (labeled_cavities == cavity_id)
        cavity_size = np.sum(cavity_mask)
        
        if cavity_size >= min_cavity_size:
            # Calculate cavity center
            cavity_coords = np.argwhere(cavity_mask)
            cavity_center = np.mean(cavity_coords, axis=0)
            
            # Convert to real coordinates
            centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
            real_center = np.array([
                centers[0][int(cavity_center[0])],
                centers[1][int(cavity_center[1])],
                centers[2][int(cavity_center[2])]
            ])
            
            # Calculate cavity volume (in voxel units)
            voxel_volume = np.prod([(edges[d][1] - edges[d][0]) for d in range(3)])
            cavity_volume = cavity_size * voxel_volume
            
            cavity_centers.append(real_center)
            cavity_volumes.append(cavity_volume)
            cavity_voxels[cavity_mask] = 1
    
    print(f"Identified {len(cavity_centers)} significant cavities")
    for i, (center, volume) in enumerate(zip(cavity_centers, cavity_volumes)):
        print(f"  Cavity {i+1}: center={center}, volume={volume:.3f}")
    
    return cavity_voxels, cavity_centers, cavity_volumes, voxel_grid, edges

def plot_clathrate_cavities(particles, shape_vertices=None, shape_color=None, 
                           cavity_threshold=0.5, min_cavity_size=10, 
                           show_particles=True, show_cavities=True,
                           particle_opacity=0.6, cavity_opacity=0.8,
                           simulation_data=None, geometry_func=None, truncation_factor=None):
    """
    Plot clathrate assembly with detected cavities highlighted.
    ...
    geometry_func: function to get geometry (default: get_bipyramid_geometry)
    truncation_factor: passed to geometry_func if not None
    """
    if geometry_func is None:
        geometry_func = get_bipyramid_geometry
    # Detect cavities
    cavity_voxels, cavity_centers, cavity_volumes, particle_voxels, edges = detect_clathrate_cavities(
        particles, shape_vertices, shape_color, 
        grid_size=64, padding=0.1, 
        cavity_threshold=cavity_threshold, 
        min_cavity_size=min_cavity_size,
        simulation_data=simulation_data
    )
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
        # Create isosurface for cavities
        z, y, x = np.mgrid[0:cavity_voxels.shape[0], 0:cavity_voxels.shape[1], 0:cavity_voxels.shape[2]]
        
        # Convert grid coordinates to real coordinates
        centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
        x_real = centers[0][x]
        y_real = centers[1][y]
        z_real = centers[2][z]
        
        # fig.add_trace(go.Isosurface(
        #     x=x_real.flatten(),
        #     y=y_real.flatten(),
        #     z=z_real.flatten(),
        #     value=cavity_voxels.flatten(),
        #     isomin=0.5,
        #     isomax=1.0,
        #     opacity=cavity_opacity,
        #     surface_count=1,
        #     colorscale='Blues',
        #     caps=dict(x_show=False, y_show=False, z_show=False),
        #     name='Cavities'
        # ))
        
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
                text=[f'Cavity {i+1}<br>Volume: {cavity_volumes[i]:.3f}'],
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
    
    return cavity_centers, cavity_volumes

def analyze_clathrate_structure(particles, shape_vertices=None, shape_color=None, 
                               simulation_data=None, geometry_func=None, truncation_factor=None):
    """
    Analyze the clathrate structure and provide detailed information about cavities.
    ...
    geometry_func: function to get geometry (default: get_bipyramid_geometry)
    truncation_factor: passed to geometry_func if not None
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
    cavity_voxels, cavity_centers, cavity_volumes, particle_voxels, edges = detect_clathrate_cavities(
        particles, shape_vertices, shape_color, simulation_data=simulation_data
    )
    total_cavity_volume = sum(cavity_volumes)
    cavity_fraction = total_cavity_volume / cell_volume
    print(f"\nCavity Analysis:")
    print(f"  Number of cavities: {len(cavity_centers)}")
    print(f"  Total cavity volume: {total_cavity_volume:.3f}")
    print(f"  Cavity fraction: {cavity_fraction:.3f} ({cavity_fraction*100:.1f}%)")
    if len(cavity_volumes) > 0:
        print(f"\nIndividual Cavity Details:")
        print(f"{'Cavity':<8} {'Volume':<12} {'Fraction':<12} {'Center':<20}")
        print("-" * 60)
        for i, (center, volume) in enumerate(zip(cavity_centers, cavity_volumes)):
            fraction = volume / cell_volume
            print(f"{i+1:<8} {volume:<12.3f} {fraction:<12.3f} {str(center):<20}")
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
        'cavity_volumes': cavity_volumes
    }

def plot_cavity_analysis(particles, shape_vertices=None, shape_color=None, 
                        simulation_data=None, geometry_func=None, truncation_factor=None):
    """
    Create comprehensive visualization of clathrate cavities with analysis.
    ...
    geometry_func: function to get geometry (default: get_bipyramid_geometry)
    truncation_factor: passed to geometry_func if not None
    """
    from plotly.subplots import make_subplots
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
    for pos, quat in particles:
        trace = get_mesh_trace(pos, quat, vertices, faces, default_color)
        if trace is not None:
            fig.add_trace(trace, row=1, col=1)
    cavity_voxels, cavity_centers, cavity_volumes, _, edges = detect_clathrate_cavities(
        particles, shape_vertices, shape_color, simulation_data=simulation_data
    )
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

#%%
# Example: Clathrate Cavity Detection and Analysis
print("\n" + "="*50)
print("CLATHRATE CAVITY DETECTION EXAMPLE")
print("="*50)

# Basic cavity detection and visualization
print("Detecting cavities in the unit cell...")
cavity_centers, cavity_volumes = plot_clathrate_cavities(
    particles=particles,
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    show_particles=True,
    show_cavities=True,
    simulation_data=simulation_data
)

# # Duplicate the unit cell
# all_particles = duplicate_unit_cell(particles, nx=3, ny=3, nz=3, simulation_data=simulation_data)
# all_positions = [(pos, quat) for pos, quat, _ in all_particles]
#%%
# Cavity detection and visualization for duplicated cells
print("Detecting cavities in the duplicated assembly...")
cavity_centers, cavity_volumes = plot_clathrate_cavities(
    particles=all_positions,
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    show_particles=True,
    show_cavities=True,
    simulation_data=simulation_data,
    geometry_func=get_bipyramid_geometry,
    min_cavity_size=5
)

#%%
print("Detecting cavities in the duplicated assembly...")
cavity_centers, cavity_volumes =   plot_clathrate_cavities(
       particles=all_positions_truncated,
       shape_vertices=shape_vertices,  # NOT truncated_vertices!
       shape_color=shape_color,
       geometry_func=get_truncated_bipyramid_geometry,
       truncation_factor=t_tomogram,
       show_particles=True,
       show_cavities=True,
       simulation_data=simulation_data,
       min_cavity_size=5
   )

#%%
# Detailed structural analysis
print("\nPerforming detailed clathrate structure analysis...")
analysis_results = analyze_clathrate_structure(
    particles=particles,
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    simulation_data=simulation_data
)


#%%
# Comprehensive visualization with analysis
print("\nCreating comprehensive cavity analysis visualization...")
plot_cavity_analysis(
    particles=particles,
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    simulation_data=simulation_data
)
#%
# Example: Show only cavities (hide particles)
print("\nVisualizing cavities only...")
plot_clathrate_cavities(
    particles=particles,
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    show_particles=False,
    show_cavities=True,
    simulation_data=simulation_data
)
#%%
print("\nClathrate cavity detection complete!")
print("Key features:")
print("- Automatically detects empty spaces between particles")
print("- Identifies individual cavity regions")
print("- Calculates cavity volumes and centers")
print("- Provides detailed structural analysis")
print("- Multiple visualization options")
















# %%



filename = 'C:\\Users\\b304014\\Software\\blee\\models\\Cages\\Cage A new.pos'  # <-- Replace with actual file path

# Parse both particles and shape definition from the same file
particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(filename)
# Get faces for the shape
_, shape_faces, _ = get_bipyramid_geometry(shape_vertices, shape_color)

# Example usage - you can modify these parameters
print("Plotting single unit cell...")
plot_particles(particles, nx=1, ny=1, nz=1, show_duplicated=False, 
                shape_vertices=shape_vertices, shape_color=shape_color, simulation_data=simulation_data)



# Example: Truncated Particle Analysis
print("\n" + "="*50)
print("TRUNCATED PARTICLE ANALYSIS")
print("="*50)

# Analyze truncation effects using analytical formulas
print("Analyzing truncation effects on volume and surface area...")
analyze_truncation_effects(truncation_factors=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Plot comparison of different truncation levels
print("\nPlotting comparison of different truncation levels...")
plot_truncation_comparison(
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    truncation_factors=[0.0, 0.3, 0.6, 1.0]
)

# Test the truncation method with different parameters
print("\nTesting truncation method...")
for t in [0.1, 0.5, 0.9]:
    print(f"\nTruncation parameter t = {t}:")
    
    # Get truncated geometry
    vertices, faces, color = get_truncated_bipyramid_geometry(shape_vertices, shape_color, t)
    print(f"  Number of vertices: {len(vertices)}")
    print(f"  Number of faces: {len(faces)}")
    
    # Calculate analytical volume and surface area
    volume, surface_area = get_truncated_bipyramid_volume_surface(t)
    print(f"  Volume: {volume:.3f}")
    print(f"  Surface area: {surface_area:.3f}")
    
    # Plot individual truncated particle
    plot_truncated_particle(
        pos=particles[0][0],
        quat=particles[0][1],
        shape_vertices=shape_vertices,
        shape_color=shape_color,
        truncation_factor=t,
        plot_type='both',
        show_axes=True
    )
#%%
# Compare assembly with different truncation levels
print("\nComparing assemblies with different truncation levels...")
for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
    print(f"Plotting assembly with truncation t = {t}...")
    plot_particles_with_truncation(
        particles=particles,
        shape_vertices=shape_vertices,
        shape_color=shape_color,
        truncation_factor=t,
        nx=1, ny=1, nz=1,
        show_duplicated=True,
        simulation_data=simulation_data
    )









#%%
print("\n" + "="*50)
print("CREATING TOMOGRAM WITH TRUNCATION")
print("="*50)

# Use moderate truncation for tomogram
t_tomogram = 0.0
print(f"Creating tomogram with truncation t = {t_tomogram}...")

# Get truncated geometry
truncated_vertices, truncated_faces, _ = get_truncated_bipyramid_geometry(
    shape_vertices, shape_color, t_tomogram
)

# Create assembly and voxelize
all_particles_truncated = duplicate_unit_cell(particles, nx=1, ny=1, nz=1, simulation_data=simulation_data)
all_positions_truncated = [(pos, quat) for pos, quat, _ in all_particles_truncated]

print("Voxelizing truncated particles...")
voxel_grid_truncated, edges_truncated = voxelize_particles(
    all_positions_truncated, 
    grid_size=128, 
    shape_vertices=truncated_vertices, 
    shape_faces=truncated_faces
)

print("Detecting cavities in the duplicated assembly...")
cavity_centers, cavity_volumes =   plot_clathrate_cavities(
       particles=all_positions_truncated,
       shape_vertices=shape_vertices,  # NOT truncated_vertices!
       shape_color=shape_color,
       geometry_func=get_truncated_bipyramid_geometry,
       truncation_factor=t_tomogram,
       show_particles=True,
       show_cavities=True,
       simulation_data=simulation_data,
       min_cavity_size=5
   )
# Detailed structural analysis
print("\nPerforming detailed clathrate structure analysis...")
analysis_results = analyze_clathrate_structure(
    particles=particles,
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    simulation_data=simulation_data
)
# Comprehensive visualization with analysis
print("\nCreating comprehensive cavity analysis visualization...")
plot_cavity_analysis(
    particles=particles,
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    simulation_data=simulation_data
)
# Example: Show only cavities (hide particles)
print("\nVisualizing cavities only...")
plot_clathrate_cavities(
    particles=particles,
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    show_particles=False,
    show_cavities=True,
    simulation_data=simulation_data
)
print("\nClathrate cavity detection complete!")
print("Key features:")
print("- Automatically detects empty spaces between particles")
print("- Identifies individual cavity regions")
print("- Calculates cavity volumes and centers")
print("- Provides detailed structural analysis")
print("- Multiple visualization options")

# %%

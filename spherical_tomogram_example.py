#!/usr/bin/env python3
"""
Spherical Particle Tomogram Example

This script demonstrates how to create and plot 3D tomograms using simple spherical particles.
This is much faster than complex geometries and useful for testing and demonstration.

Author: Generated Script
Date: Current
"""

import numpy as np
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from clathrate_analysis import parse_particles_and_shape, duplicate_unit_cell
from tomogram_utils import (
    save_voxel_grid_as_tiff,
    plot_3d_tomogram,
    plot_3d_isosurface,
    plot_3d_volume,
    plot_tomogram_slices_2d
)

def create_spherical_voxel_grid(particles, grid_size=64, padding=0.1, sphere_radius=0.2):
    """
    Create a voxel grid with spherical particles instead of complex geometries.
    
    Args:
        particles: List of (position, quaternion) tuples
        grid_size: Number of voxels per dimension
        padding: Fraction of box size to pad on each side
        sphere_radius: Radius of each spherical particle
    
    Returns:
        voxel_grid: 3D numpy array
        edges: Grid edges for reference
    """
    print(f"Creating spherical voxel grid with {len(particles)} particles...")
    print(f"Sphere radius: {sphere_radius}")
    
    # Get particle positions (ignore orientations for spheres)
    positions = np.array([pos for pos, _ in particles])
    
    # Calculate bounding box
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
    
    # Calculate voxel size
    voxel_size = (edges[0][1] - edges[0][0])
    print(f"Voxel size: {voxel_size:.4f}")
    print(f"Grid bounds: [{min_corner[0]:.2f}, {max_corner[0]:.2f}] × [{min_corner[1]:.2f}, {max_corner[1]:.2f}] × [{min_corner[2]:.2f}, {max_corner[2]:.2f}]")
    
    # Create coordinate grids for distance calculation
    z_grid, y_grid, x_grid = np.meshgrid(
        np.arange(grid_size), 
        np.arange(grid_size), 
        np.arange(grid_size), 
        indexing='ij'
    )
    
    # Convert grid indices to real coordinates
    x_coords = centers[0][x_grid]
    y_coords = centers[1][y_grid]
    z_coords = centers[2][z_grid]
    
    # Fill in spherical particles
    particles_filled = 0
    for i, (pos, _) in enumerate(particles):
        if i % 50 == 0:  # Progress indicator
            print(f"Processing particle {i+1}/{len(particles)}")
        
        # Calculate distance from each voxel to particle center
        distances = np.sqrt(
            (x_coords - pos[0])**2 + 
            (y_coords - pos[1])**2 + 
            (z_coords - pos[2])**2
        )
        
        # Mark voxels within sphere radius
        sphere_mask = distances <= sphere_radius
        voxel_grid[sphere_mask] = 1.0
        
        if np.any(sphere_mask):
            particles_filled += 1
    
    filled_voxels = np.sum(voxel_grid > 0)
    print(f"Filled {filled_voxels} voxels from {particles_filled} particles")
    print(f"Fill fraction: {filled_voxels / voxel_grid.size:.4f}")
    
    return voxel_grid, edges

def create_spherical_tomogram_with_cavities(particles, grid_size=64, padding=0.1, 
                                          sphere_radius=0.2, cavity_radius=0.15):
    """
    Create a tomogram with spherical particles and spherical cavity objects.
    
    Args:
        particles: List of (position, quaternion) tuples  
        grid_size: Number of voxels per dimension
        padding: Fraction of box size to pad on each side
        sphere_radius: Radius of particle spheres
        cavity_radius: Radius of cavity spheres
    
    Returns:
        voxel_grid: 3D numpy array with particles (1.0) and cavities (2.0)
        edges: Grid edges for reference
    """
    print("Creating tomogram with spherical particles and cavity objects...")
    
    # First create the particle grid
    voxel_grid, edges = create_spherical_voxel_grid(particles, grid_size, padding, sphere_radius)
    
    # Find potential cavity locations (empty spaces)
    empty_mask = (voxel_grid == 0)
    
    # Simple cavity detection: find large empty regions
    from scipy import ndimage
    
    # Use distance transform to find centers of empty regions
    distance_transform = ndimage.distance_transform_edt(empty_mask)
    
    # Find local maxima that are sufficiently large
    min_cavity_distance = cavity_radius / ((edges[0][1] - edges[0][0]))  # Convert to voxel units
    potential_cavities = distance_transform >= min_cavity_distance
    
    # Use peak detection to find separated cavity centers
    from scipy.ndimage import maximum_filter
    neighborhood_size = int(min_cavity_distance * 2)
    local_maxima = maximum_filter(distance_transform, size=neighborhood_size)
    cavity_centers_mask = (distance_transform == local_maxima) & potential_cavities
    
    # Get cavity center coordinates
    cavity_indices = np.argwhere(cavity_centers_mask)
    print(f"Found {len(cavity_indices)} potential cavity centers")
    
    if len(cavity_indices) == 0:
        print("No cavities found")
        return voxel_grid, edges
    
    # Limit number of cavities for demonstration
    max_cavities = 10
    if len(cavity_indices) > max_cavities:
        # Sort by distance value and take the largest
        distances_at_centers = [distance_transform[tuple(idx)] for idx in cavity_indices]
        sorted_indices = sorted(range(len(cavity_indices)), 
                              key=lambda i: distances_at_centers[i], reverse=True)
        cavity_indices = cavity_indices[sorted_indices[:max_cavities]]
        print(f"Limited to {max_cavities} largest cavities")
    
    # Convert grid coordinates to real coordinates  
    centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
    
    # Create coordinate grids for cavity placement
    z_grid, y_grid, x_grid = np.meshgrid(
        np.arange(grid_size), 
        np.arange(grid_size), 
        np.arange(grid_size), 
        indexing='ij'
    )
    x_coords = centers[0][x_grid]
    y_coords = centers[1][y_grid]  
    z_coords = centers[2][z_grid]
    
    # Place spherical cavity objects
    cavities_placed = 0
    for cavity_idx in cavity_indices:
        # Get real coordinates of cavity center
        cavity_center = [
            centers[0][cavity_idx[0]],
            centers[1][cavity_idx[1]], 
            centers[2][cavity_idx[2]]
        ]
        
        # Calculate distances to cavity center
        distances = np.sqrt(
            (x_coords - cavity_center[0])**2 + 
            (y_coords - cavity_center[1])**2 + 
            (z_coords - cavity_center[2])**2
        )
        
        # Mark cavity object voxels (value 2.0)
        cavity_mask = distances <= cavity_radius
        
        # Only place cavity objects in empty space
        empty_cavity_mask = cavity_mask & (voxel_grid == 0)
        voxel_grid[empty_cavity_mask] = 2.0
        
        if np.any(empty_cavity_mask):
            cavities_placed += 1
            print(f"Placed cavity {cavities_placed} at {cavity_center}")
    
    cavity_voxels = np.sum(voxel_grid == 2.0)
    print(f"Placed {cavities_placed} cavity objects using {cavity_voxels} voxels")
    
    return voxel_grid, edges

def example_spherical_particles_basic(particle_file=None):
    """
    Example 1: Basic spherical particle tomogram
    """
    print("="*60)
    print("SPHERICAL PARTICLES: Basic Tomogram")
    print("="*60)
    
    # Load or create particle data
    if particle_file and os.path.exists(particle_file):
        print(f"Loading particles from: {particle_file}")
        particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(particle_file)
    else:
        print("Creating synthetic spherical particle data...")
        particles, simulation_data = create_synthetic_spherical_particles()
    
    print(f"Using {len(particles)} particles")
    
    # Create spherical voxel grid
    sphere_radius = 0.15  # Smaller radius for better resolution
    voxel_grid, edges = create_spherical_voxel_grid(
        particles=particles,
        grid_size=128,  # Higher resolution for better spheres
        padding=0.1,
        sphere_radius=sphere_radius
    )
    
    # Save as TIFF
    filename = "spherical_particles_tomogram.tif"
    save_voxel_grid_as_tiff(voxel_grid, filename, pixel_size=1.0)
    print(f"Saved spherical tomogram: {filename}")
    
    # Visualize
    print("\nCreating isosurface visualization...")
    plot_3d_tomogram(filename, plot_type='isosurface', threshold=0.5, colorscale='Viridis')
    
    print("\nCreating volume rendering...")
    plot_3d_tomogram(filename, plot_type='volume', opacity=0.3, colorscale='Blues')
    
    return filename, particles

def example_spherical_particles_with_cavities(particles):
    """
    Example 2: Spherical particles with spherical cavity objects
    """
    print("\n" + "="*60)
    print("SPHERICAL PARTICLES: With Cavity Objects")
    print("="*60)
    
    # Create tomogram with cavity objects
    voxel_grid, edges = create_spherical_tomogram_with_cavities(
        particles=particles,
        grid_size=128,
        padding=0.1,
        sphere_radius=0.15,  # Particle radius
        cavity_radius=0.08   # Smaller cavity objects
    )
    
    # Save as TIFF
    filename = "spherical_particles_with_cavities.tif"
    save_voxel_grid_as_tiff(voxel_grid, filename, pixel_size=1.0)
    print(f"Saved cavity tomogram: {filename}")
    
    # Analyze the result
    particle_voxels = np.sum(voxel_grid == 1.0)
    cavity_voxels = np.sum(voxel_grid == 2.0)
    empty_voxels = np.sum(voxel_grid == 0.0)
    total_voxels = voxel_grid.size
    
    print(f"\nTomogram Analysis:")
    print(f"  Total voxels: {total_voxels}")
    print(f"  Particle voxels: {particle_voxels} ({particle_voxels/total_voxels*100:.1f}%)")
    print(f"  Cavity voxels: {cavity_voxels} ({cavity_voxels/total_voxels*100:.1f}%)")
    print(f"  Empty voxels: {empty_voxels} ({empty_voxels/total_voxels*100:.1f}%)")
    
    # Visualize
    print("\nCreating isosurface visualization...")
    plot_3d_tomogram(filename, plot_type='isosurface', threshold=0.5, colorscale='Plasma')
    
    print("\nCreating 2D slice views...")
    plot_tomogram_slices_2d(filename, slice_type='middle')
    
    return filename

def example_spherical_particles_large_assembly(particle_file=None):
    """
    Example 3: Large assembly of spherical particles using unit cell duplication
    """
    print("\n" + "="*60)
    print("SPHERICAL PARTICLES: Large Assembly")
    print("="*60)
    
    # Load or create base particle data
    if particle_file and os.path.exists(particle_file):
        print(f"Loading base particles from: {particle_file}")
        particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(particle_file)
    else:
        print("Creating synthetic base particle data...")
        particles, simulation_data = create_synthetic_spherical_particles(n_base=2)  # Smaller base for duplication
    
    print(f"Base unit cell has {len(particles)} particles")
    
    # Duplicate the unit cell to create a larger assembly
    print("Duplicating unit cell to create large assembly...")
    duplicated_particles = duplicate_unit_cell(particles, nx=3, ny=3, nz=3, simulation_data=simulation_data)
    large_particles = [(pos, quat) for pos, quat, _ in duplicated_particles]
    
    print(f"Large assembly has {len(large_particles)} particles")
    
    # Create spherical voxel grid for large assembly
    voxel_grid, edges = create_spherical_voxel_grid(
        particles=large_particles,
        grid_size=128,  # Keep reasonable for memory
        padding=0.05,   # Less padding for large assembly
        sphere_radius=0.12  # Slightly smaller spheres
    )
    
    # Save as TIFF
    filename = "large_spherical_assembly.tif"
    save_voxel_grid_as_tiff(voxel_grid, filename, pixel_size=1.0)
    print(f"Saved large assembly tomogram: {filename}")
    
    # Visualize
    print("\nCreating isosurface visualization...")
    plot_3d_tomogram(filename, plot_type='isosurface', threshold=0.5, colorscale='Viridis')
    
    return filename

def create_synthetic_spherical_particles(n_base=4):
    """
    Create synthetic particle data optimized for spherical representation.
    
    Args:
        n_base: Base dimension for creating n_base^3 particles
    
    Returns:
        particles: List of (position, quaternion) tuples
        simulation_data: None (no simulation data for synthetic)
    """
    print(f"Creating {n_base}³ = {n_base**3} synthetic particles...")
    
    particles = []
    spacing = 1.0  # Spacing between particle centers
    
    for i in range(n_base):
        for j in range(n_base):
            for k in range(n_base):
                # Position particles in a regular grid
                x = i * spacing - (n_base-1) * spacing / 2
                y = j * spacing - (n_base-1) * spacing / 2
                z = k * spacing - (n_base-1) * spacing / 2
                
                # Add small random displacement
                x += np.random.uniform(-0.1, 0.1)
                y += np.random.uniform(-0.1, 0.1)
                z += np.random.uniform(-0.1, 0.1)
                
                # Orientation doesn't matter for spheres, but keep for compatibility
                quat = (1, 0, 0, 0)  # No rotation
                
                particles.append(((x, y, z), quat))
    
    print(f"Created {len(particles)} particles in grid arrangement")
    return particles, None

def main():
    """
    Main function to run spherical particle tomogram examples
    """
    print("="*60)
    print("SPHERICAL PARTICLE TOMOGRAM EXAMPLES")
    print("="*60)
    
    # You can specify a particle file here
    particle_file = None  # Set to your .pos file path if available
    
    try:
        # Example 1: Basic spherical particles
        basic_file, particles = example_spherical_particles_basic(particle_file)
        
        # Example 2: Spherical particles with cavity objects
        cavity_file = example_spherical_particles_with_cavities(particles)
        
        # Example 3: Large assembly of spherical particles
        large_file = example_spherical_particles_large_assembly(particle_file)
        
        print("\n" + "="*60)
        print("SPHERICAL TOMOGRAM EXAMPLES COMPLETE")
        print("="*60)
        print("\nGenerated files:")
        for filename in [basic_file, cavity_file, large_file]:
            if filename and os.path.exists(filename):
                print(f"  - {filename}")
        
        print("\nAdvantages of spherical particles:")
        print("  - Much faster computation")
        print("  - Simpler geometry calculations")
        print("  - Good for testing and prototyping")
        print("  - Easier to understand results")
        print("  - Less memory usage")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Make sure you have the required dependencies installed:")
        print("  pip install numpy plotly scipy tifffile")
        raise

if __name__ == "__main__":
    main()

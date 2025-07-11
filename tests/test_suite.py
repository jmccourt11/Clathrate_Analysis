"""
Comprehensive Test Suite for Clathrate Analysis

This module consolidates all testing functionality for the clathrate analysis package.
It includes tests for cavity detection, center detection, and debugging utilities.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the modules
import clathrate_analysis
importlib.reload(clathrate_analysis)

from clathrate_analysis import (
    parse_particles_and_shape, voxelize_particles, detect_clathrate_cavities,
    get_truncated_bipyramid_geometry, plot_cavity_spheres, get_bipyramid_geometry,
    get_mesh_trace, duplicate_unit_cell
)

class TestCavityDetection:
    """Test class for cavity detection functionality."""
    
    def __init__(self, filename=None):
        """
        Initialize test suite with a particle data file.
        
        Args:
            filename: Path to particle data file (optional)
        """
        self.filename = filename or 'C:\\Users\\b304014\\Software\\blee\\models\\Cages\\Cage A new.pos'
        self.particles = None
        self.shape_vertices = None
        self.shape_color = None
        self.simulation_data = None
        
    def load_data(self):
        """Load particle data for testing."""
        print(f"Loading file: {self.filename}")
        self.particles, self.shape_vertices, self.shape_color, self.simulation_data = parse_particles_and_shape(self.filename)
        print(f"Found {len(self.particles)} particles")
        
        # Create duplicated particles
        duplicated_particles = duplicate_unit_cell(self.particles, nx=1, ny=1, nz=1, simulation_data=self.simulation_data)
        self.duplicated_positions = [(pos, quat) for pos, quat, _ in duplicated_particles]
        print(f"Duplicated to {len(self.duplicated_positions)} particles")
        
    def test_cavity_detection_step_by_step(self, truncation_factor=0.5, grid_size=64, 
                                         padding=0.1, min_cavity_size=10):
        """
        Test cavity detection step by step to debug issues.
        
        Args:
            truncation_factor: Truncation factor for particles
            grid_size: Grid size for voxelization
            padding: Padding for voxelization
            min_cavity_size: Minimum cavity size threshold
        """
        if not self.particles:
            self.load_data()
            
        print(f"\nTesting with:")
        print(f"  Truncation factor: {truncation_factor}")
        print(f"  Grid size: {grid_size}")
        print(f"  Padding: {padding}")
        print(f"  Min cavity size: {min_cavity_size}")
        
        # Step 1: Test voxelization
        print("\n" + "="*50)
        print("STEP 1: Testing voxelization")
        print("="*50)
        
        try:
            voxel_grid, edges = voxelize_particles(
                self.duplicated_positions,
                grid_size=grid_size,
                padding=padding,
                shape_vertices=self.shape_vertices,
                geometry_func=get_truncated_bipyramid_geometry,
                truncation_factor=truncation_factor
            )
            
            print(f"Voxel grid shape: {voxel_grid.shape}")
            print(f"Total voxels: {voxel_grid.size}")
            print(f"Particle voxels: {np.sum(voxel_grid)}")
            print(f"Empty voxels: {voxel_grid.size - np.sum(voxel_grid)}")
            print(f"Particle fraction: {np.sum(voxel_grid) / voxel_grid.size:.3f}")
            
            # Visualize the voxel grid
            print("\nVisualizing voxel grid...")
            z, y, x = np.mgrid[0:voxel_grid.shape[0], 0:voxel_grid.shape[1], 0:voxel_grid.shape[2]]
            centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
            x_real = centers[0][x]
            y_real = centers[1][y]
            z_real = centers[2][z]
            
            fig = go.Figure()
            fig.add_trace(go.Isosurface(
                x=x_real.flatten(),
                y=y_real.flatten(),
                z=z_real.flatten(),
                value=voxel_grid.flatten(),
                isomin=0.5,
                isomax=1.0,
                opacity=0.6,
                surface_count=1,
                colorscale='Blues',
                caps=dict(x_show=False, y_show=False, z_show=False),
                name='Particles'
            ))
            
            fig.update_layout(
                title=f'Voxelized Particles (Grid {grid_size}, Truncation {truncation_factor})',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=40)
            )
            fig.show()
            
        except Exception as e:
            print(f"Error in voxelization: {e}")
            return
        
        # Step 2: Test cavity detection
        print("\n" + "="*50)
        print("STEP 2: Testing cavity detection")
        print("="*50)
        
        try:
            cavity_voxels, cavity_centers, cavity_volumes, cavity_radii, particle_voxels, edges = detect_clathrate_cavities(
                particles=self.duplicated_positions,
                shape_vertices=self.shape_vertices,
                shape_color=self.shape_color,
                grid_size=grid_size,
                padding=padding,
                min_cavity_size=min_cavity_size,
                simulation_data=self.simulation_data,
                geometry_func=get_truncated_bipyramid_geometry,
                truncation_factor=truncation_factor,
                keep_largest_cavity_only=False
            )
            
            print(f"Found {len(cavity_centers)} cavities")
            
            if len(cavity_centers) > 0:
                print("\nCavity details:")
                for i, (center, volume, radius) in enumerate(zip(cavity_centers, cavity_volumes, cavity_radii)):
                    print(f"  Cavity {i+1}: center={center}, volume={volume:.6f}, radius={radius:.6f}")
                
                # Visualize cavities
                print("\nVisualizing cavities...")
                z, y, x = np.mgrid[0:cavity_voxels.shape[0], 0:cavity_voxels.shape[1], 0:cavity_voxels.shape[2]]
                centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
                x_real = centers[0][x]
                y_real = centers[1][y]
                z_real = centers[2][z]
                
                fig = go.Figure()
                
                # Add particles
                fig.add_trace(go.Isosurface(
                    x=x_real.flatten(),
                    y=y_real.flatten(),
                    z=z_real.flatten(),
                    value=particle_voxels.flatten(),
                    isomin=0.5,
                    isomax=1.0,
                    opacity=0.3,
                    surface_count=1,
                    colorscale='Blues',
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    name='Particles'
                ))
                
                # Add cavities
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
                    name='Cavities'
                ))
                
                # Add cavity centers
                for i, center in enumerate(cavity_centers):
                    fig.add_trace(go.Scatter3d(
                        x=[center[0]],
                        y=[center[1]],
                        z=[center[2]],
                        mode='markers',
                        marker=dict(size=10, color='yellow'),
                        name=f'Cavity {i+1} Center'
                    ))
                
                fig.update_layout(
                    title=f'Cavities Found: {len(cavity_centers)} (Grid {grid_size}, Truncation {truncation_factor})',
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z',
                        aspectmode='data'
                    ),
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                fig.show()
                
            else:
                print("No cavities detected!")
                
        except Exception as e:
            print(f"Error in cavity detection: {e}")
            
    def test_cavity_center_detection(self, truncation_factor=0.4, grid_size=64, 
                                   padding=0.1, min_cavity_size=10):
        """
        Test that cavity centers are detected at optimal locations.
        
        Args:
            truncation_factor: Truncation factor for particles
            grid_size: Grid size for voxelization
            padding: Padding for voxelization
            min_cavity_size: Minimum cavity size threshold
        """
        if not self.particles:
            self.load_data()
            
        print(f"\nTesting cavity center detection with:")
        print(f"  Truncation factor: {truncation_factor}")
        print(f"  Grid size: {grid_size}")
        print(f"  Padding: {padding}")
        print(f"  Min cavity size: {min_cavity_size}")
        
        # Detect cavities
        cavity_voxels, cavity_centers, cavity_volumes, cavity_radii, voxel_grid, edges = detect_clathrate_cavities(
            particles=self.duplicated_positions,
            shape_vertices=self.shape_vertices,
            shape_color=self.shape_color,
            grid_size=grid_size,
            padding=padding,
            min_cavity_size=min_cavity_size,
            simulation_data=self.simulation_data,
            geometry_func=get_truncated_bipyramid_geometry,
            truncation_factor=truncation_factor,
            keep_largest_cavity_only=False
        )
        
        print(f"\nFound {len(cavity_centers)} cavities")
        
        if len(cavity_centers) > 0:
            # Calculate the geometric center of all particles for comparison
            particle_positions = np.array([pos for pos, _ in self.duplicated_positions])
            geometric_center = np.mean(particle_positions, axis=0)
            
            print(f"\nGeometric center of all particles: {geometric_center}")
            
            for i, (center, volume, radius) in enumerate(zip(cavity_centers, cavity_volumes, cavity_radii)):
                print(f"\nCavity {i+1}:")
                print(f"  Detected center: {center}")
                print(f"  Distance from geometric center: {np.linalg.norm(center - geometric_center):.6f}")
                print(f"  Volume: {volume:.6f}")
                print(f"  Radius: {radius:.6f}")
            
            # Visualize the results
            print("\nCreating visualization...")
            
            fig = go.Figure()
            
            # Plot particles
            vertices, faces, default_color = get_truncated_bipyramid_geometry(self.shape_vertices, self.shape_color, truncation_factor)
            for pos, quat in self.duplicated_positions:
                trace = get_mesh_trace(pos, quat, vertices, faces, default_color)
                if trace is not None:
                    fig.add_trace(trace)
            
            # Plot cavity centers
            for i, center in enumerate(cavity_centers):
                fig.add_trace(go.Scatter3d(
                    x=[center[0]],
                    y=[center[1]],
                    z=[center[2]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='sphere'
                    ),
                    name=f'Cavity {i+1} Center (r={cavity_radii[i]:.4f})'
                ))
            
            # Plot geometric center for comparison
            fig.add_trace(go.Scatter3d(
                x=[geometric_center[0]],
                y=[geometric_center[1]],
                z=[geometric_center[2]],
                mode='markers',
                marker=dict(
                    size=10,
                    color='yellow',
                    symbol='diamond'
                ),
                name='Geometric Center'
            ))
            
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
                    opacity=0.3,
                    colorscale='Reds',
                    showscale=False,
                    name=f'Cavity {i+1} Sphere'
                ))
            
            fig.update_layout(
                title=f'Cavity Centers vs Geometric Center - {len(cavity_centers)} cavities detected',
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
            
            # Check if centers are significantly different from geometric center
            print(f"\nAnalysis:")
            for i, center in enumerate(cavity_centers):
                distance = np.linalg.norm(center - geometric_center)
                if distance > 0.1:  # Threshold for "significantly different"
                    print(f"  Cavity {i+1}: Center is significantly different from geometric center (distance: {distance:.4f})")
                else:
                    print(f"  Cavity {i+1}: Center is close to geometric center (distance: {distance:.4f})")
            
        else:
            print("No cavities detected!")
            
    def debug_cavity_detection(self, truncation_factor=0.4, grid_sizes=[64, 128], 
                             min_cavity_sizes=[10, 50, 100], padding_values=[0.1, 0.2]):
        """
        Debug cavity detection with different parameters.
        
        Args:
            truncation_factor: Truncation factor for particles
            grid_sizes: List of grid sizes to test
            min_cavity_sizes: List of minimum cavity sizes to test
            padding_values: List of padding values to test
        """
        if not self.particles:
            self.load_data()
            
        print(f"Number of particles: {len(self.duplicated_positions)}")
        
        # Test different parameter combinations
        results = []
        
        for grid_size in grid_sizes:
            for min_cavity_size in min_cavity_sizes:
                for padding in padding_values:
                    print(f"\n" + "="*60)
                    print(f"Testing: grid_size={grid_size}, min_cavity_size={min_cavity_size}, padding={padding}")
                    print("="*60)
                    
                    try:
                        # Detect cavities
                        cavity_voxels, cavity_centers, cavity_volumes, cavity_radii, voxel_grid, edges = detect_clathrate_cavities(
                            particles=self.duplicated_positions,
                            shape_vertices=self.shape_vertices,
                            shape_color=self.shape_color,
                            grid_size=grid_size,
                            padding=padding,
                            min_cavity_size=min_cavity_size,
                            simulation_data=self.simulation_data,
                            geometry_func=get_truncated_bipyramid_geometry,
                            truncation_factor=truncation_factor,
                            keep_largest_cavity_only=False
                        )
                        
                        result = {
                            'grid_size': grid_size,
                            'min_cavity_size': min_cavity_size,
                            'padding': padding,
                            'num_cavities': len(cavity_centers),
                            'cavity_centers': cavity_centers,
                            'cavity_volumes': cavity_volumes,
                            'cavity_radii': cavity_radii,
                            'voxel_grid': voxel_grid,
                            'cavity_voxels': cavity_voxels,
                            'edges': edges
                        }
                        
                        results.append(result)
                        
                        print(f"Found {len(cavity_centers)} cavities")
                        if len(cavity_radii) > 0:
                            print(f"Average radius: {np.mean(cavity_radii):.4f}")
                            print(f"Max radius: {np.max(cavity_radii):.4f}")
                            print(f"Total volume: {np.sum(cavity_volumes):.6f}")
                        
                    except Exception as e:
                        print(f"Error with parameters: {e}")
                        continue
        
        return results
    
    def visualize_cavity_comparison(self, results):
        """
        Create a comparison visualization of cavity detection results.
        
        Args:
            results: List of cavity detection results
        """
        if not results:
            print("No results to visualize")
            return
        
        # Create subplots
        n_results = len(results)
        cols = min(3, n_results)
        rows = (n_results + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Grid={r['grid_size']}, Min={r['min_cavity_size']}, Pad={r['padding']}\n"
                           f"Cavities: {r['num_cavities']}" for r in results],
            specs=[[{'type': 'mesh3d'} for _ in range(cols)] for _ in range(rows)]
        )
        
        for i, result in enumerate(results):
            row = i // cols + 1
            col = i % cols + 1
            
            if result['num_cavities'] > 0:
                # Plot cavity voxels as isosurface
                cavity_voxels = result['cavity_voxels']
                edges = result['edges']
                
                # Create coordinate grids
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
                    opacity=0.6,
                    surface_count=1,
                    colorscale='Reds',
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    showlegend=False
                ), row=row, col=col)
        
        fig.update_layout(
            title="Cavity Detection Comparison",
            height=300 * rows,
            showlegend=False
        )
        
        fig.show()

def run_all_tests(filename=None):
    """
    Run all tests with default parameters.
    
    Args:
        filename: Path to particle data file (optional)
    """
    print("="*80)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Initialize test suite
    test_suite = TestCavityDetection(filename)
    
    # Run step-by-step cavity detection test
    print("\n1. Testing cavity detection step by step...")
    test_suite.test_cavity_detection_step_by_step()
    
    # Run cavity center detection test
    print("\n2. Testing cavity center detection...")
    test_suite.test_cavity_center_detection()
    
    # Run debugging with multiple parameters
    print("\n3. Running cavity detection debugging...")
    results = test_suite.debug_cavity_detection()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Grid':<6} {'Min':<6} {'Pad':<6} {'Cavities':<10} {'Avg Radius':<12} {'Max Radius':<12}")
    print("-" * 80)
    
    for result in results:
        if result['num_cavities'] > 0:
            avg_radius = np.mean(result['cavity_radii'])
            max_radius = np.max(result['cavity_radii'])
        else:
            avg_radius = 0
            max_radius = 0
            
        print(f"{result['grid_size']:<6} {result['min_cavity_size']:<6} {result['padding']:<6} "
              f"{result['num_cavities']:<10} {avg_radius:<12.4f} {max_radius:<12.4f}")
    
    # Find the best result
    if results:
        best_result = max(results, key=lambda r: (r['num_cavities'], np.mean(r['cavity_radii']) if r['cavity_radii'] else 0))
        print(f"\nBest result: Grid={best_result['grid_size']}, Min={best_result['min_cavity_size']}, Pad={best_result['padding']}")
        print(f"Found {best_result['num_cavities']} cavities")
        
        # Visualize the best result
        if best_result['num_cavities'] > 0:
            print("\nVisualizing best result...")
            test_suite.visualize_cavity_comparison([best_result])
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_all_tests() 
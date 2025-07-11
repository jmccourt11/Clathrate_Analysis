#%%
"""
Truncated Structure Cavity Analysis Example

This script demonstrates how to perform cavity analysis on truncated triangular bipyramids.
It shows how cavity properties change with different truncation levels.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clathrate_analysis import (
    parse_particles_and_shape, plot_clathrate_cavities, analyze_clathrate_structure,
    plot_cavity_analysis, get_truncated_bipyramid_geometry
)

def main():
    """Main function demonstrating truncated cavity analysis."""
    
    print("="*70)
    print("TRUNCATED STRUCTURE CAVITY ANALYSIS")
    print("="*70)
    
    # Load particle data
    filename = 'C:\\Users\\b304014\\Software\\blee\\models\\Cages\\Cage A new.pos'
    print(f"\n1. Loading particle data from: {filename}")
    
    particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(filename)
    print(f"   Loaded {len(particles)} particles")
    
    # Test different truncation levels
    truncation_factors = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    print(f"\n2. Analyzing cavities for different truncation levels...")
    print("-" * 60)
    
    results = {}
    
    for truncation_factor in truncation_factors:
        print(f"\n   Truncation t = {truncation_factor}:")
        print("   " + "-" * 40)
        
        # Detect cavities with truncated geometry
        cavity_centers, cavity_volumes, cavity_radii = plot_clathrate_cavities(
            particles=particles,
            shape_vertices=shape_vertices,
            shape_color=shape_color,
            geometry_func=get_truncated_bipyramid_geometry,
            truncation_factor=truncation_factor,
            show_particles=True,
            show_cavities=True,
            show_spheres=True,
            simulation_data=simulation_data
        )
        
        # Perform detailed analysis
        analysis_results = analyze_clathrate_structure(
            particles=particles,
            shape_vertices=shape_vertices,
            shape_color=shape_color,
            geometry_func=get_truncated_bipyramid_geometry,
            truncation_factor=truncation_factor,
            simulation_data=simulation_data
        )
        
        # Store results
        results[truncation_factor] = {
            'cavity_centers': cavity_centers,
            'cavity_volumes': cavity_volumes,
            'cavity_radii': cavity_radii,
            'analysis': analysis_results
        }
        
        print(f"   Summary:")
        print(f"     - Cavities found: {len(cavity_centers)}")
        print(f"     - Total volume: {sum(cavity_volumes):.6f}")
        print(f"     - Average radius: {np.mean(cavity_radii):.6f}")
        print(f"     - Cavity fraction: {analysis_results['cavity_fraction']:.6f}")
    
    # Create summary comparison
    print(f"\n3. Summary Comparison:")
    print("=" * 60)
    print(f"{'t':<6} {'Cavities':<10} {'Total Vol':<12} {'Avg Radius':<12} {'Cavity %':<10}")
    print("-" * 60)
    
    for t in truncation_factors:
        result = results[t]
        cavity_count = len(result['cavity_centers'])
        total_vol = sum(result['cavity_volumes'])
        avg_radius = np.mean(result['cavity_radii']) if result['cavity_radii'] else 0
        cavity_pct = result['analysis']['cavity_fraction'] * 100
        
        print(f"{t:<6.1f} {cavity_count:<10} {total_vol:<12.6f} {avg_radius:<12.6f} {cavity_pct:<10.2f}")
    
    # Find optimal truncation for maximum cavity volume
    max_volume_trunc = max(truncation_factors, 
                          key=lambda t: sum(results[t]['cavity_volumes']))
    max_volume = sum(results[max_volume_trunc]['cavity_volumes'])
    
    print(f"\n4. Optimal Truncation Analysis:")
    print("-" * 40)
    print(f"   Maximum cavity volume occurs at t = {max_volume_trunc}")
    print(f"   Maximum total cavity volume: {max_volume:.6f}")
    
    # Show detailed analysis for optimal truncation
    print(f"\n5. Detailed Analysis for Optimal Truncation (t = {max_volume_trunc}):")
    print("-" * 60)
    
    optimal_results = results[max_volume_trunc]
    plot_cavity_analysis(
        particles=particles,
        shape_vertices=shape_vertices,
        shape_color=shape_color,
        geometry_func=get_truncated_bipyramid_geometry,
        truncation_factor=max_volume_trunc,
        simulation_data=simulation_data
    )
    
    print(f"\n" + "="*70)
    print("TRUNCATED CAVITY ANALYSIS COMPLETE")
    print("="*70)
    
    return results

if __name__ == "__main__":
    results = main()
# %% 
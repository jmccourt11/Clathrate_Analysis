#!/usr/bin/env python3
"""
Main Entry Point for Clathrate Analysis

This script provides easy access to the clathrate analysis package functionality.
It can be used to run the GUI, examples, or tests.

Usage:
    python main.py [command] [options]

Commands:
    gui         - Launch the Streamlit GUI
    example     - Run the example usage script
    test        - Run the test suite
    help        - Show this help message
"""

import sys
import os
import subprocess
import argparse

def run_gui():
    """Launch the Streamlit GUI."""
    print("Launching Clathrate Analysis GUI...")
    gui_path = os.path.join("src", "clathrate_gui.py")
    subprocess.run(["streamlit", "run", gui_path])

def run_example():
    """Run the example usage script."""
    print("Running example usage...")
    example_path = os.path.join("examples", "example_usage.py")
    subprocess.run([sys.executable, example_path])

def run_test():
    """Run the test suite."""
    print("Running test suite...")
    test_path = os.path.join("tests", "test_suite.py")
    subprocess.run([sys.executable, test_path])

def show_help():
    """Show help information."""
    print(__doc__)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clathrate Analysis Package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "command",
        choices=["gui", "example", "test", "help"],
        help="Command to run"
    )
    
    args = parser.parse_args()
    
    if args.command == "gui":
        run_gui()
    elif args.command == "example":
        run_example()
    elif args.command == "test":
        run_test()
    elif args.command == "help":
        show_help()

if __name__ == "__main__":
    main() 
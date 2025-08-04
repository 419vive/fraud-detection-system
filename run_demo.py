#!/usr/bin/env python3
"""
IEEE-CIS Fraud Detection Demo Launcher
=====================================

This script provides easy setup and launch of the fraud detection demo application.
It handles environment setup, dependency checking, and application startup.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import argparse
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DemoLauncher:
    """Demo application launcher and setup manager"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "demo_requirements.txt"
        self.demo_app = self.project_root / "demo_app.py"
        self.config_file = self.project_root / "demo_config.json"
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        logger.info(f"Python version: {sys.version}")
        return True
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        try:
            import streamlit
            import pandas
            import numpy
            import plotly
            logger.info("Core dependencies are available")
            return True
        except ImportError as e:
            logger.warning(f"Missing dependency: {e}")
            return False
    
    def install_dependencies(self, force: bool = False):
        """Install required dependencies"""
        if not force and self.check_dependencies():
            logger.info("Dependencies already satisfied")
            return True
        
        if not self.requirements_file.exists():
            logger.error(f"Requirements file not found: {self.requirements_file}")
            return False
        
        logger.info("Installing dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ])
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def setup_demo_data(self):
        """Set up sample data for the demo"""
        try:
            from demo_utils import create_sample_datasets, generate_sample_config, save_demo_config
            
            logger.info("Creating sample datasets...")
            create_sample_datasets()
            
            # Create demo configuration if it doesn't exist
            if not self.config_file.exists():
                logger.info("Creating demo configuration...")
                config = generate_sample_config()
                save_demo_config(config, str(self.config_file))
            
            logger.info("Demo data setup completed")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import demo utilities: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to setup demo data: {e}")
            return False
    
    def check_app_files(self):
        """Check if required application files exist"""
        required_files = [
            self.demo_app,
            self.project_root / "demo_utils.py"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        logger.info("All required application files are present")
        return True
    
    def launch_streamlit(self, port: int = 8501, host: str = "localhost"):
        """Launch the Streamlit application"""
        if not self.demo_app.exists():
            logger.error(f"Demo app not found: {self.demo_app}")
            return False
        
        logger.info(f"Starting Streamlit app on {host}:{port}")
        logger.info(f"App will be available at: http://{host}:{port}")
        
        try:
            subprocess.run([
                "streamlit", "run", str(self.demo_app),
                "--server.port", str(port),
                "--server.address", host,
                "--server.headless", "false",
                "--browser.gatherUsageStats", "false"
            ])
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Streamlit: {e}")
            return False
        except FileNotFoundError:
            logger.error("Streamlit not found. Please install it using: pip install streamlit")
            return False
        
        return True
    
    def launch_prediction_service(self, port: int = 8000):
        """Launch the FastAPI prediction service"""
        prediction_service = self.project_root / "src" / "prediction_service.py"
        
        if not prediction_service.exists():
            logger.warning("Prediction service not found. Demo will run in simulation mode.")
            return True
        
        logger.info(f"Starting FastAPI prediction service on port {port}")
        
        try:
            subprocess.Popen([
                sys.executable, str(prediction_service)
            ])
            logger.info(f"Prediction service available at: http://localhost:{port}")
            return True
        except Exception as e:
            logger.warning(f"Could not start prediction service: {e}")
            logger.info("Demo will run in simulation mode")
            return True
    
    def run_full_setup(self, force_install: bool = False):
        """Run complete setup process"""
        logger.info("=== IEEE-CIS Fraud Detection Demo Setup ===")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Check application files
        if not self.check_app_files():
            return False
        
        # Install dependencies
        if not self.install_dependencies(force_install):
            return False
        
        # Setup demo data
        if not self.setup_demo_data():
            return False
        
        logger.info("=== Setup completed successfully ===")
        return True
    
    def launch_demo(self, port: int = 8501, host: str = "localhost", 
                   with_api: bool = False, api_port: int = 8000):
        """Launch the complete demo environment"""
        
        # Launch prediction service if requested
        if with_api:
            self.launch_prediction_service(api_port)
        
        # Launch Streamlit app
        self.launch_streamlit(port, host)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="IEEE-CIS Fraud Detection Demo Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py --setup                    # Setup only
  python run_demo.py --launch                   # Launch demo app
  python run_demo.py --setup --launch           # Setup and launch
  python run_demo.py --launch --port 8502       # Launch on custom port
  python run_demo.py --launch --with-api        # Launch with API service
        """
    )
    
    parser.add_argument("--setup", action="store_true",
                       help="Run setup process (install dependencies, create sample data)")
    parser.add_argument("--launch", action="store_true",
                       help="Launch the demo application")
    parser.add_argument("--force-install", action="store_true",
                       help="Force reinstallation of dependencies")
    parser.add_argument("--port", type=int, default=8501,
                       help="Port for Streamlit app (default: 8501)")
    parser.add_argument("--host", default="localhost",
                       help="Host for Streamlit app (default: localhost)")
    parser.add_argument("--with-api", action="store_true",
                       help="Also launch the FastAPI prediction service")
    parser.add_argument("--api-port", type=int, default=8000,
                       help="Port for API service (default: 8000)")
    
    args = parser.parse_args()
    
    launcher = DemoLauncher()
    
    # If no arguments provided, do setup and launch
    if not any([args.setup, args.launch]):
        args.setup = True
        args.launch = True
    
    success = True
    
    # Run setup if requested
    if args.setup:
        success = launcher.run_full_setup(args.force_install)
    
    # Launch demo if requested and setup was successful
    if args.launch and success:
        launcher.launch_demo(
            port=args.port,
            host=args.host,
            with_api=args.with_api,
            api_port=args.api_port
        )
    elif args.launch and not success:
        logger.error("Setup failed. Cannot launch demo.")
        sys.exit(1)

if __name__ == "__main__":
    main()
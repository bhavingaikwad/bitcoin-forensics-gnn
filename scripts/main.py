import subprocess
import sys
import logging

# Standard logging configuration for production scripts
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def exec_pipeline():
    """Sequentially executes the forensic toolchain."""
    
    scripts = [
        "forensic_feature_analysis.py",
        "topology_visualization.py",
        "benchmark_evaluation.py",
        "class_balancing_pipeline.py",
        "gnn_inference_engine.py"
    ]

    for script in scripts:
        logger.info(f"Running module: {script}")
        try:
            # shell=False is a security best practice to prevent injection
            subprocess.run([sys.executable, f"scripts/{script}"], check=True)
        except subprocess.CalledProcessError:
            logger.error(f"Pipeline failed at {script}")
            sys.exit(1)

    logger.info("Pipeline execution finalized.")

if __name__ == "__main__":
    exec_pipeline()
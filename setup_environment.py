import subprocess
import sys


def is_cuda_available():
    """Check if CUDA is available on the system."""
    try:
        subprocess.run(
            ["nvidia-smi"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True
    except Exception:
        return False


def create_conda_environment(env_file_path):
    """Create a Conda environment from a yml file."""
    try:
        subprocess.run(["conda", "env", "create", "-f", env_file_path], check=True)
        print("Conda environment created successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to create Conda environment:", e)
        sys.exit(1)


def install_cuda_packages(env_name):
    """Install CUDA-dependent packages in the specified Conda environment."""
    cuda_packages = ["cupy-cuda11x"]

    for package in cuda_packages:
        cmd = f"conda run -n {env_name} pip install {package}"
        try:
            subprocess.run(cmd, check=True, shell=True)
            print(f"Installed {package} in {env_name} environment.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package} in {env_name}: {e}")
            sys.exit(1)


def main():
    env_file_path = "environment.yml"

    # Create Conda environment
    create_conda_environment(env_file_path)

    # Check for CUDA availability and install packages if necessary
    if is_cuda_available():
        print("CUDA is available. Installing CUDA-dependent packages.")
        install_cuda_packages("neurometry")
    else:
        print("CUDA not available. Skipping CUDA-dependent packages.")


if __name__ == "__main__":
    main()

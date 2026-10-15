#!/usr/bin/env bash
# Installer for the grasp-synthesis worker env ("bodex") + Isaac Lab sim assets.
#
#   bash scripts/install.sh                 # env "bodex", PyTorch cu128 (RTX 30/40/50 series)
#   ENV_NAME=myenv TORCH_CUDA=cu121 bash scripts/install.sh
#   SKIP_SIM_ASSETS=1 bash scripts/install.sh   # hardware-only machine
#
# Prerequisites: NVIDIA driver >= 570, CUDA toolkit 12.8 (nvcc on PATH), conda, git, a C++17
# compiler. Isaac Sim 5.1 + Isaac Lab (separate env) for simulation, ROS 2 Humble for hardware.
set -eo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-bodex}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
REL="https://github.com/ismael-esp/dexterous-pick-and-place/releases/download/v1.0"

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
cd "$ROOT"

echo "==> [1/5] PyTorch ($TORCH_CUDA) + conda-forge deps (coal collision library)"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
conda install -y -c conda-forge coal==3.0.1 eigen=3.4

echo "==> [2/5] cuRobo/BODex package (builds the CUDA kernels in src/curobo/curobolib/cpp)"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
pip install -e . --no-build-isolation
pip install usd-core
pip install --no-build-isolation torch-scatter
pip install "numpy==1.26.4"                    # newer NumPy breaks warp / torch-scatter

echo "==> [3/5] coal_openmp_wrapper (second native extension)"
(cd src/curobo/geom/cpp && python setup.py install)

echo "==> [4/5] Environment hooks"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/bodex_env.sh" <<EOF
# conda's newer libstdc++ (CXXABI_1.3.15) must win over the system one for coal
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
# the Isaac Lab env and the ROS runner spawn the worker with this interpreter
export BODEX_SERVICE_PYTHON="\$CONDA_PREFIX/bin/python"
EOF

echo "==> [5/5] Isaac Lab scene assets (T-shape torso, UR3/UR3e+Allegro USDs, object USDs)"
if [ -z "$SKIP_SIM_ASSETS" ] && [ ! -d isaaclab_tasks/assets ]; then
  curl -L "$REL/isaaclab_assets.tar.zst" | zstd -d | tar -x -C isaaclab_tasks
fi

echo
echo "Done. Next:  conda activate $ENV_NAME && python scripts/check_install.py"
echo "Simulation also needs an Isaac Lab env (tested: Isaac Sim 5.1 + Isaac Lab 2.3.0): see README > Installation."
echo "Set BODEX_SERVICE_PYTHON=$CONDA_PREFIX/bin/python in that env (or your shell) too."

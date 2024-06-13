# Install dependencies
pip3 install -r requirements.txt --no-cache-dir

# Check if zett is already installed
if [ -d "zett" ]; then
    echo "Zett is already installed."
else
    echo "Installing Zett..."
    cd zett
    pip3 install -r requirements.txt --no-cache-dir
    pip3 install -U --no-cache-dir "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # adjust based on your CUDA version
    pip3 install -e .
fi

# Check if FVT is already installed
if [ -d "fast-vocabulary-transfer" ]; then
    echo "FVT is already installed."
fi
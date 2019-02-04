# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
export PATH=/tools/local/bin:/tools/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/tools/local/lib:/tools/cuda-9.0/lib64:/tools/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/tools/cuda-9.0

# Tools config for CUDA, Anaconda installed in the common /tools directory
cd /storage/home/karthikt/SparseExplanations
source /scratch/scratch4/karthikt/envs/magic35/bin/activate
python -u scripts/classifier_sparse.py &> out

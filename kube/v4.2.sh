# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
export PATH=/tools/local/bin:/tools/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/tools/local/lib:/tools/cuda-9.0/lib64:/tools/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/tools/cuda-9.0

# Tools config for CUDA, Anaconda installed in the common /tools directory
cd /storage/home/karthikt/XAI
source /scratch/scratch4/karthikt/envs/magic35/bin/activate
python -u models/v4.2/classifier_train.py --epochs 25 --lr 0.001 --batch_size 32 --num_maps 8 --patience 5 --limit 3 --save_model 'models/v4.2/classifier' &> models/v4.2/log_train
python -u models/v4.2/classifier_test.py --batch_size 10 --num_maps 8 --load_model 'models/v4.2/classifier' --save_results 'models/v4.2/results' &> models/v4.2/log_test
python -u models/v4.2/visualize.py --load_model 'models/v4.2/classifier' --plot_folder 'models/v4.2/plots' --num_maps 8 &> models/v4.2/log_vizualize

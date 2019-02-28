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
python -u models/v1/autoencoder_train.py --epochs 15 --lr 0.001 --batch_size 20 --num_maps 3 --patience 10 --limit 4 --save_model 'models/v1/autoencoder' &> models/v1/log_auto_train
python -u models/v1/autoencoder_test.py --batch_size 100 --num_maps 3 --load_model 'models/v1/autoencoder' --save_results '/scratch/scratch2/karthikt/data/feats.h5' &> models/v1/log_auto_test
python -u models/v1/autoencoder_visualize.py --num_maps 3 --test_file '/scratch/scratch2/karthikt/data/feats.h5' --plot_folder 'models/v1/recon' &> models/v1/log_auto_visuzlize 

# The project uses two paths that need to be customized.
# The first one is "YOUR_ROOT": save the experiment path to access it from other files.
# The second one is "PATH_ENCODER": This is the path that points to your offline trained encoder.
# For the sake of simplicity, leave everything the same and only adjust the path to your root directory via "logdir
import os
import datetime


EXPERIMENT_NAME = "DDHRL_Test_Training"
root = "/home/shawan/Desktop/Shawan/ddhRL"

logdir = root + '/results'
path_experiment = "run_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
YOUR_ROOT= os.path.join(logdir, path_experiment)

PATH_ENCODER =  root + "/offline_trained_encoder/r18_byol.onnx"

# Training parameters
NUM_GPUS=2
NUM_CPUS=7
NUM_ITERATIONS = 10000  # 500 results in Tensorboard shown with 500 iterations (about an hour)
NUM_AGENTS = 2
GPU_ID = '0,1' # Cuda visible device/gpu

# In case you want to restore form checkpoint
RESUME = False
RESTORE_PATH =""

# Save experiment path to access it from other files
import os
import datetime

logdir = '/home/shawan/Desktop/Shawan/ddhRL/results'
path_experiment = "run_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
YOUR_ROOT= os.path.join(logdir, path_experiment)

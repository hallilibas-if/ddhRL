from tensorboardX import SummaryWriter
import numpy as np
import tensorflow as tf

class tensorboard_writer():
    def __init__(self, configuration):
        self.writer = SummaryWriter(logdir="../Distribution/")
        # For logging actions mean an variance
        self.writer_freq = 1
        self.writer_current_step = 0
        self.configuration = configuration

    def write(self, logits):
        # if enough data is available to compute mean and variance
        if logits.shape[0] > 32:
            # print("Training logits:", logits.shape)
            if self.configuration["OUTPUTS"] == 1:  # only steering
                try:
                    self.writer.add_histogram('Steering mean', logits[:, 0].numpy(), self.writer_current_step)
                    self.writer.add_histogram('Steering variance', tf.exp(logits[:, 1]).numpy(), self.writer_current_step)
                    #print("Stuff added to summary writer: ", logits[0, 0], logits[0, 1])
                except:
                    print("Warning: Nothing added to summary writer")
            elif self.configuration["OUTPUTS"] == 2:  # steering and acceleration
                mean, std = tf.split(logits, 2, axis=1)
                #mean, std, raw_scale_perturb_factor = tf.split(logits, [self.configuration["OUTPUTS"], self.configuration["OUTPUTS"], self.configuration["OUTPUTS"] * (self.configuration["OUTPUTS"] - 1) // 2], axis=1)
                steering_mean = mean[:, 0].numpy()
                throttle_mean = mean[:, 1].numpy()
                steering_std = np.exp((std[:, 0]).numpy())
                throttle_std = np.exp((std[:, 1]).numpy())

                try:
                    self.writer.add_histogram('Steering action mean', steering_mean, self.writer_current_step)
                    self.writer.add_histogram('Steering action variance', steering_std,
                                              self.writer_current_step)
                    self.writer.add_histogram('Throttle action mean', throttle_mean, self.writer_current_step)
                    self.writer.add_histogram('Throttle action variance', throttle_std,
                                              self.writer_current_step)
                except:
                    print("Warning: Nothing added to summary writer")
            elif self.configuration["OUTPUTS"] == 3:  # steering, braking and throttle
                mean, std = tf.split(logits, 2, axis=1)
                steering_mean = mean[:, 0].numpy()
                throttle_mean = mean[:, 1].numpy()
                break_mean = mean[:, 2].numpy()
                steering_std = tf.exp(std[:, 0]).numpy()
                throttle_std = tf.exp(std[:, 1]).numpy()
                break_std = tf.exp(std[:, 2]).numpy()
                try:
                    self.writer.add_histogram('Steering action mean', steering_mean, self.writer_current_step)
                    self.writer.add_histogram('Steering action variance', steering_std,
                                              self.writer_current_step)
                    self.writer.add_histogram('Throttle action mean', throttle_mean, self.writer_current_step)
                    self.writer.add_histogram('Throttle action variance', throttle_std,
                                              self.writer_current_step)
                    self.writer.add_histogram('Break action mean', break_mean, self.writer_current_step)
                    self.writer.add_histogram('Break action variance', break_std, self.writer_current_step)
                except:
                    print("Warning: Nothing added to summary writer")

            self.writer_current_step += 1
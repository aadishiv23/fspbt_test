import tensorflow as tf
import os
import shutil

class Logger(object):
    def __init__(self, log_dir, suffix=None):
        """Create a summary writer logging to log_dir."""
        if suffix:
            log_dir = os.path.join(log_dir, suffix)
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()



class ModelLogger(object):
    def __init__(self, log_dir, save_func):
        self.log_dir = log_dir
        self.save_func = save_func

    def save(self, model, epoch, isGenerator):
        if isGenerator:
            new_path = os.path.join(self.log_dir, "model_%05d.pth" % epoch)
        else:
            new_path = os.path.join(self.log_dir, "disc_%05d.pth" % epoch)
        self.save_func(model, new_path)

    def copy_file(self, source):
        shutil.copy(source, self.log_dir)

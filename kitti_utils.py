import pykitti

# load data set using pykitti

class KITTIdata():
    """

    ignore the first image for differing

    """
    def __init__(self, basedir, sequences):
        self.sequence_nos = sequences

        self.dataset = {}
        for sequence in sequences:
            self.dataset[sequence] =  pykitti.odometry(basedir, sequence)

    def _process(self, images):
        # images is a tuple = (prev_image, curr__image)
        # take diff with previous image per channel = gives 3 new channels
        # append this to for 150x50x6 from two 150x50x3
        # return new_image
        pass

    def get_series_batch(self, sequence_len = 100,  batch_size = 10, sequences = None):
        # will provide series from all kitti sequences, unless specified
        if sequences is None:
            pass
        else:
            pass

    def get_series(self, sequence_len = 100, sequences = None):
        pass















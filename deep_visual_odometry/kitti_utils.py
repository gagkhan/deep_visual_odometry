import pykitti
import numpy as np
# load data set using pykitti

class KITTIdata():
    """
    ignore the first image for differing

    """
    def __init__(self, basedir, sequences, img_size = None):
        self.sequences = sequences
        self.img_size = img_size
        self.dataset = {}
        self.dataset_len = {}
        self.img_idx = {}
        self.seq_idx = 0
        for sequence in sequences:
            self.dataset[sequence] =  pykitti.odometry(basedir, sequence)
            self.dataset_len[sequence] = len(self.dataset[sequence].cam2_files)
            self.img_idx[sequence] = 0

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
        if sequences is None:
           sequences = self.sequences
        while self.sequences[self.seq_idx] not in sequences:
            self.seq_idx += 1
        sequence = self.sequences[self.seq_idx]

        # input
        images = []
        for i in range(sequence_len + 1):
            image = self.dataset[sequence].get_cam2(i + self.img_idx[sequence])
            if self.img_size is not None:
                image = image.resize(size=self.img_size)
            image = np.array(image, dtype=np.uint8)
            images.append(image)

        image_plus_diff_images = []
        for i in range(sequence_len):
            diff_image = images[i+1] - images[i]
            image_plus_diff_images.append(np.concatenate((images[i], diff_image), axis = 2))
        series_input = np.stack(image_plus_diff_images)

        # TODO: ground truth poses
        self.img_idx[sequence] += 1
        if self.img_idx[sequence] + sequence_len >= self.dataset_len[sequence] - 1:
            self.img_idx[sequence] = 0
        while self.sequences[self.seq_idx] not in sequences:
            self.seq_idx+= 1

        #TODO: return ground truth poses
        return series_input

























import pykitti
import numpy as np
# load data set using pykitti

def angle(T):
    R = T[:3, :3]
    return np.arctan2(R[1, 0], R[0, 0])

def get_vel(T_prev, T_curr):
    delta_T = np.dot(np.linalg.inv(T_prev), T_curr)
    delta_rot = angle(delta_T)
    delta_pos = np.linalg.norm(delta_T[0:3, 3])
    return np.array([delta_pos, delta_rot])

def get_pose(T):
    x = T[0, 3]
    y = T[1, 3]
    theta = angle(T)
    return np.array([x, y, theta])

class KITTIdata(object):
    """
    as
    dsa

    """
    def __init__(self, basedir, sequences, img_size = None):
        self.sequences = sequences
        self.img_size = img_size

        self.dataset = {}
        self.dataset_len = {}

        self.img_idx = {}
        self.seq_idx = 0

        self.input = {}
        self.velocities = {}
        self.poses = {}

        for sequence in sequences:

            dataset = pykitti.odometry(basedir, sequence)
            self.dataset[sequence] =  dataset
            self.dataset_len[sequence] = len(self.dataset[sequence].cam2_files)
            self.img_idx[sequence] = 0

            input_images = []
            velocities = []
            poses = []

            image_prev = dataset.get_cam2(0)
            if self.img_size is not None:
                image_prev = image_prev.resize(size = self.img_size)
            image_prev = np.array(image_prev, dtype=np.uint8)

            for i in range(1, self.dataset_len[sequence]):
                image = dataset.get_cam2(i)
                if self.img_size is not None:
                    image = image.resize(size=self.img_size)
                image = np.array(image, dtype=np.uint8)
                diff_image = image - image_prev
                input_images.append(np.concatenate((image, diff_image), axis = 2))
                image_prev = image
                velocities.append(get_vel(dataset.poses[i-1], dataset.poses[i]))
                poses.append(get_pose(dataset.poses[i]))

            self.input[sequence] = np.stack(input_images)
            self.velocities[sequence] = np.stack(velocities)
            self.poses[sequence] = np.stack(poses)
            print('completed load sequence {} data'.format(sequence))

    def get_series_batch(self, sequence_len = 100,  batch_size = 10, sequences = None):
        batch_input = []
        batch_velocities = []
        batch_poses = []
        for _ in range(batch_size):
            input_images, velocities, poses = self.get_series(sequence_len)
            batch_input.append(input_images)
            batch_velocities.append(velocities)
            batch_poses.append(poses)

        return np.stack(batch_input), np.stack(velocities), np.stack(poses)

    def get_series(self, sequence_len = 100, sequences = None, ):
        if sequences is None:
           sequences = self.sequences
        self.seq_idx %= len(self.sequences)
        while self.sequences[self.seq_idx] not in sequences:
            self.seq_idx += 1

        sequence = self.sequences[self.seq_idx]
        idx = self.img_idx[sequence]
        series_input = self.input[sequence][idx: idx + sequence_len]
        velocities = self.velocities[sequence][idx: idx + sequence_len]
        poses = self.poses[sequence][idx: idx + sequence_len]

        self.img_idx[sequence] += 1
        if self.img_idx[sequence] + sequence_len >= self.dataset_len[sequence] - 1:
            self.img_idx[sequence] = 0
        self.seq_idx+= 1

        return series_input, velocities, poses


    def load_data_input_model(self):
        input_images = []
        velocities = []
        for sequence in self.sequences:
            input_images.append(self.input[sequence])
            velocities.append(self.velocities[sequence])
        input_images = np.concatenate(input_images)
        velocities = np.concatenate(velocities)
        return input_images, velocities



































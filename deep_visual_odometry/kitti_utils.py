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

    def get_series_batch(self, sequence_len = 100,  batch_size = 10, sequences = None):
        batch_input = []
        batch_velocities = []
        batch_poses = []
        for _ in range(batch_size):
            input, velocities, poses = self.get_series(sequence_len)
            batch_input.append(input)
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
            image_plus_diff_images.append(np.concatenate((images[i+1], diff_image), axis = 2))
        series_input = np.stack(image_plus_diff_images)

        # def normalize_cols(M):
        #     for i in range(M.shape[1]):
        #         M[:, i] /= np.linalg.norm(M[:, i])
        #     return M

        def angle(T):
            R = T[:3, :3]
            return np.arctan2(R[1, 0], R[0, 0])

        def get_vel(T_prev, T_curr):
            delta_T = np.dot(np.linalg.inv(T_prev), T_curr)
            delta_rot = angle(delta_T)
            delta_pos = np.linalg.norm(delta_T[0:3, 3])
            return np.array([delta_pos, delta_rot])

        def get_pose(T):
            x = T[0,3]
            y = T[1,3]
            theta = angle(T)
            return np.array([x, y, theta])

        # poses and estimated velocity
        poses = []
        velocities = []
        dataset_poses = self.dataset[sequence].poses
        for i in range(sequence_len):
            velocities.append(get_vel(dataset_poses[i],dataset_poses[i+1]))
            poses.append(get_pose(dataset_poses[i+1]))
        poses = np.stack(poses)
        velocities = np.stack(velocities)

        self.img_idx[sequence] += 1
        if self.img_idx[sequence] + sequence_len >= self.dataset_len[sequence] - 1:
            self.img_idx[sequence] = 0
        self.seq_idx+= 1

        return series_input, velocities, poses

































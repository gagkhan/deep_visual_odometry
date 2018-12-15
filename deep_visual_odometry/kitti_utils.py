import pykitti
import numpy as np
# load data set using pykitti

def angle(T):
    R = T[:3, :3]
    return np.arctan2(R[1, 0], R[0, 0])

def get_vel(T_prev, T_curr):
    # delta_T = np.dot(np.linalg.inv(T_prev), T_curr)
    # delta_rot = angle(delta_T)
    # delta_pos = np.linalg.norm(delta_T[0:3, 3])

    delta_pose = get_pose(T_curr) - get_pose(T_prev)
    delta_pos = np.linalg.norm(delta_pose[0:2])
    delta_rot = delta_pose[2]
    while delta_rot >= np.pi:
        delta_rot -= 2*np.pi
    while delta_rot < -np.pi:
        delta_rot += 2*np.pi

    return np.array([delta_pos, delta_rot])

def get_pose(T):
    x = T[0, 3]
    y = T[1, 3]
    theta = angle(T)
    return np.array([x, y, theta])

class KITTIdata(object):
    """


    """
    def __init__(self, basedir, sequences,sequence_len=100, val_frac = 0.1,test_frac = 0.1, img_size = None):
        self.sequences = sequences
        self.img_size = img_size
        self.sequence_len = sequence_len

        self.dataset = {}
        self.dataset_len = {}

        self.img_idx = {}
        self.seq_idx = 0

        self.input = {}
        self.velocities = {}
        self.prev_poses = {}
        self.target_poses = {}

        self.masks = {}
        self.train_mask = {}
        self.val_mask = {}
        self.test_mask = {}

        
        self.train_idx = {}
        self.val_idx = {}
        self.test_idx = {}
        self.val_frac = val_frac
        self.test_frac = test_frac
        
        for sequence in sequences:

            dataset = pykitti.odometry(basedir, sequence)
            self.dataset[sequence] =  dataset
            self.dataset_len[sequence] = len(self.dataset[sequence].cam2_files)
            self.img_idx[sequence] = 0

            # mask definition
            np.random.seed(100)
            self.masks[sequence] = np.random.choice(self.dataset_len[sequence]-sequence_len,
                                                    self.dataset_len[sequence]-sequence_len,
                                                    replace = False)

            mask_len = self.dataset_len[sequence]-sequence_len

            self.train_mask[sequence] = self.masks[sequence][0:int(np.round(mask_len*(1-val_frac-test_frac)))]
            self.val_mask[sequence] = self.masks[sequence][int(np.round(mask_len*(1-val_frac-test_frac))):
                                                           int(np.round(mask_len*(1-test_frac)))]
            self.test_mask[sequence] = self.masks[sequence][int(np.round(mask_len*(1-test_frac))):mask_len]
            self.train_idx[sequence] = 0
            self.val_idx[sequence] = 0
            self.test_idx[sequence] = 0
            
            input_images = []
            velocities = []
            prev_poses = []
            target_poses = []

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
                target_poses.append(get_pose(dataset.poses[i]))
                prev_poses.append(get_pose(dataset.poses[i-1]))
                velocities.append(get_vel(dataset.poses[i-1], dataset.poses[i]))

            self.input[sequence] = np.stack(input_images)
            self.velocities[sequence] = np.stack(velocities)
            self.prev_poses[sequence] = np.stack(prev_poses)
            self.target_poses[sequence] = np.stack(target_poses)

            print('completed load sequence {} data'.format(sequence))

    def normalize(self):
        all_inputs = np.vstack([self.input[_] for _ in self.sequences])
        self.input_mean = np.mean(all_inputs, axis = 0)
        self.input_std = np.std(all_inputs, axis = 0)
        
        all_poses = np.vstack([self.prev_poses[_] for _ in self.sequences])
        self.pose_mean = np.mean(all_poses, axis = 0)
        self.pose_std = np.std(all_poses, axis = 0)
        
        all_velocities = np.vstack([self.velocities[_] for _ in self.sequences])
        self.velocities_mean = np.mean(all_velocities, axis = 0)
        self.velocities_std = np.std(all_velocities, axis = 0)
        
        for sequence in self.sequences:
            self.input[sequence] = (self.input[sequence] - self.input_mean)/self.input_std
            self.prev_poses[sequence] = (self.prev_poses[sequence] - self.pose_mean) / self.pose_std
            self.target_poses[sequence] = (self.target_poses[sequence] - self.pose_mean) / self.pose_std
            self.velocities[sequence] = (self.velocities[sequence] - self.velocities_mean) / self.velocities_std

        print('normalized data')
        
    def get_series_train(self, sequences = None):
        if sequences is None:
           sequences = self.sequences
        sequence_len = self.sequence_len
        sequence = sequences[np.random.randint(0, len(sequences))]

        train_mask = self.train_mask[sequence]
        train_mask_len = len(train_mask)
        train_idx = self.train_idx[sequence]
        sequence_range = range(train_mask[train_idx], train_mask[train_idx] + sequence_len)

        input_ = self.input[sequence][sequence_range]
        inter_input = np.hstack([self.velocities[sequence][sequence_range],
                                 self.prev_poses[sequence][sequence_range]])
        target = self.target_poses[sequence][sequence_range]

        self.train_idx[sequence] += 1
        if self.train_idx[sequence] >= train_mask_len-1:
            self.train_idx[sequence] = 0
        return input_, inter_input, target

    def get_series_batch_train(self, batch_size = 10, sequences = None):
        batch_inputs = []
        batch_inter_inputs = []
        batch_targets = []

        if sequences is None:
            sequences = self.sequences

        for _ in range(batch_size):
            inputs, inter_inputs, targets = self.get_series_train(sequences = sequences)
            batch_inputs.append(inputs)
            batch_inter_inputs.append(inter_inputs)
            batch_targets.append(targets)

        return np.stack(batch_inputs), np.stack(batch_inter_inputs), np.stack(batch_targets)


    def load_all_sequences(self, mode, sequences = None):

        inputs = []
        inter_inputs = []
        targets = []
        sequence_len = self.sequence_len
        if sequences is None:
            sequences = self.sequences

        if mode == 'train':
            idx_dict = self.train_idx
            mask_dict = self.train_mask
        elif mode == 'val':
            idx_dict = self.val_idx
            mask_dict = self.val_mask
        elif mode == 'test':
            idx_dict = self.test_idx
            mask_dict = self.test_mask
        else:
            ValueError('unknown mode')

        for sequence in sequences:
            mask = mask_dict[sequence]
            idx = idx_dict[sequence]
            while idx < len(mask):
                sequence_range = range(mask[idx], mask[idx] + sequence_len)
                input_ = self.input[sequence][sequence_range]
                velocities = self.velocities[sequence][sequence_range]
                prev_poses = self.prev_poses[sequence][sequence_range]
                target_poses = self.target_poses[sequence][sequence_range]
                idx += 1
                inputs.append(input_)
                inter_inputs.append(np.hstack([velocities, prev_poses]))
                targets.append(target_poses)

        return np.stack(inputs), np.stack(inter_inputs), np.stack(targets)

    def load_data_input_model(self):
        input_images = []
        velocities = []
        for sequence in self.sequences:
            input_images.append(self.input[sequence])
            velocities.append(self.velocities[sequence])
        input_images = np.concatenate(input_images)
        velocities = np.concatenate(velocities)
        
        '''
        # augment dataset by adding a flipped dataset across Y-axis
        images_flipped = np.flip(input_images[:,:,:,0:3],axis = 2)
        diff_flipped = np.flip(input_images[:,:,:,3:6],axis = 2)
        input_images_flipped = np.concatenate((images_flipped,diff_flipped),axis = 3)
        input_images_final = np.concatenate((input_images,input_images_flipped),axis = 0)
        
        velocities_flipped = np.zeros_like(velocities)
        velocities_flipped[:,0]=velocities[:,0]*(-1)
        velocities_flipped[:,1]=velocities[:,1]
        velocities_final = np.concatenate((velocities,velocities_flipped),axis = 0)
        # add noise to some percentage of the images
        
        indices = np.random.choice(input_images.shape[0],int(0.3*input_images.shape[0]))
        input_images[indices] = input_images[indices]+5*np.random.normal(0,0.7)
        '''
        return input_images, velocities

    def get_full_sequence(self, sequence):
        inputs = self.input[sequence]
        inter_inputs = np.hstack(self.velocities[sequence], self.prev_poses[sequence])
        targets = self.target_poses[sequence]
        return inputs, inter_inputs, targets
































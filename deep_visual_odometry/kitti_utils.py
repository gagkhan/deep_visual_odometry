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
    def __init__(self, basedir, sequences,sequence_len=100,val_frac = 0.1,test_frac = 0.1, img_size = None):
        self.sequences = sequences
        self.img_size = img_size

        self.dataset = {}
        self.dataset_len = {}

        self.img_idx = {}
        self.seq_idx = 0

        self.input = {}
        
        self.velocities = {}
        self.poses = {}
        
        self.masks = {}
        self.index_train = {}
        self.index_val = {}
        self.index_test = {}
        
        for sequence in sequences:

            dataset = pykitti.odometry(basedir, sequence)
            self.dataset[sequence] =  dataset
            self.dataset_len[sequence] = len(self.dataset[sequence].cam2_files)
            self.img_idx[sequence] = 0
            self.index_val[sequence] = int(np.floor((self.dataset_len[sequence]-sequence_len)*(1-val_frac-test_frac)))
            self.index_test[sequence] = int(np.floor((self.dataset_len[sequence]-sequence_len)*(1-val_frac-test_frac) + (self.dataset_len[sequence]-sequence_len)*val_frac))
            self.index_train[sequence] = 0
            # mask definition
            np.random.seed(100)
            self.masks[sequence] = np.random.choice(self.dataset_len[sequence]-sequence_len,self.dataset_len[sequence]-sequence_len,replace = False)

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
        print(idx)
        series_input = self.input[sequence][idx: idx + sequence_len]
        velocities = self.velocities[sequence][idx: idx + sequence_len]
        poses = self.poses[sequence][idx: idx + sequence_len]

        self.img_idx[sequence] += 1
        if self.img_idx[sequence] + sequence_len >= self.dataset_len[sequence] - 1:
            self.img_idx[sequence] = 0
        self.seq_idx+= 1

        return series_input, velocities, poses

    # new functions to load data
    def get_series_batch_train(self, batch_size = 10, sequence_len = 100,val_frac = 0.1,test_frac = 0.1, sequences = None, ):
        batch_input = []
        batch_velocities = []
        batch_poses = []
        for _ in range(batch_size):
            
            input_images, velocities, poses = self.get_series_train(sequence_len,val_frac,test_frac,mode = 'train')
            batch_input.append(input_images)
            batch_velocities.append(velocities)
            batch_poses.append(poses)

        return np.stack(batch_input), np.stack(velocities), np.stack(poses)

    def load_data_validation(self,sequence_len = 100, val_frac = 0.1, test_frac = 0.1, sequences = None,):
        val_input = []
        val_velocities = []
        val_poses = []
        if sequences is None:
           sequences = self.sequences
        for sequence in sequences:
            seq_len = self.dataset_len[sequence]
            index_mask = self.masks[sequence]
            val_idx = self.index_val[sequence]
            print(val_idx)
            test_portion = int(np.floor((self.dataset_len[sequence]-sequence_len)*test_frac))
            print(index_mask[val_idx],test_portion)
            while (index_mask[val_idx]+sequence_len < seq_len-test_portion-1):
                series_input = self.input[sequence][index_mask[val_idx]:index_mask[val_idx]+sequence_len]
                velocities = self.velocities[sequence][index_mask[val_idx]:index_mask[val_idx]+sequence_len]
                poses = self.poses[sequence][index_mask[val_idx]:index_mask[val_idx]+sequence_len]
                val_idx+=1
                print(val_idx,sequence)
                val_input.append(series_input)
                val_velocities.append(velocities)
                val_poses.append(poses)

        return np.stack(val_input),np.stack(val_velocities),np.stack(val_poses)

    def load_data_test(self,sequence_len = 100, val_frac = 0.1, test_frac = 0.1, sequences = None,):
        test_input = []
        test_velocities = []
        test_poses = []
        if sequences is None:
           sequences = self.sequences
        for sequence in sequences:
            seq_len = self.dataset_len[sequence]
            index_mask = self.masks[sequence]
            test_idx = self.index_test[sequence]
            print(test_idx)
            print(index_mask[test_idx])
            while (test_idx < len(index_mask) and(index_mask[test_idx]+sequence_len) < seq_len-1):
                series_input = self.input[sequence][index_mask[test_idx]:index_mask[test_idx]+sequence_len]
                velocities = self.velocities[sequence][index_mask[test_idx]:index_mask[test_idx]+sequence_len]
                poses = self.poses[sequence][index_mask[test_idx]:index_mask[test_idx]+sequence_len]
                test_idx+=1
                print(test_idx,sequence)
                test_input.append(series_input)
                test_velocities.append(velocities)
                test_poses.append(poses)

        return np.stack(test_input),np.stack(test_velocities),np.stack(test_poses)
    
    def get_series_train(self, sequence_len = 100, val_frac = 0.1, test_frac = 0.1, mode = 'train',sequences = None,):
        if sequences is None:
           sequences = self.sequences
        seq_num = np.random.randint(0,len(sequences))
        selected_sequence = sequences[seq_num]
        index_mask = self.masks[selected_sequence]
        seq_len = self.dataset_len[selected_sequence]
        
        if mode == 'train':
            train_idx = self.index_train[selected_sequence]
            print(train_idx,index_mask[train_idx]+sequence_len)
            
            series_input = self.input[selected_sequence][index_mask[train_idx]:index_mask[train_idx]+sequence_len]
            velocities = self.velocities[selected_sequence][index_mask[train_idx]:index_mask[train_idx]+sequence_len]
            poses = self.poses[selected_sequence][index_mask[train_idx]:index_mask[train_idx]+sequence_len]
            self.index_train[selected_sequence]+=1
            
    
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



































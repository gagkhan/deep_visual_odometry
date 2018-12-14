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
    as
    dsa

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
        self.poses = {}
        
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
            self.masks[sequence] = np.random.choice(self.dataset_len[sequence]-sequence_len-1,self.dataset_len[sequence]-sequence_len-1,replace = False)
            mask_len = self.dataset_len[sequence]-sequence_len
    
            self.train_mask[sequence] = self.masks[sequence][0:int(np.round(mask_len*(1-val_frac-test_frac)))]
            self.val_mask[sequence] = self.masks[sequence][int(np.round(mask_len*(1-val_frac-test_frac))):int(np.round(mask_len*(1-test_frac)))]
            self.test_mask[sequence] = self.masks[sequence][int(np.round(mask_len*(1-test_frac))):mask_len]

            self.train_idx[sequence] = 0
            self.val_idx[sequence] = 0
            self.test_idx[sequence] = 0
            
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
                poses.append(get_pose(dataset.poses[i]))
                velocities.append(get_vel(dataset.poses[i-1], dataset.poses[i]))

            self.input[sequence] = np.stack(input_images)
            self.velocities[sequence] = np.stack(velocities)
            self.poses[sequence] = np.stack(poses)
            print('completed load sequence {} data'.format(sequence))
            


    def get_series_batch(self, batch_size = 10, sequences = None):
        batch_input = []
        batch_velocities = []
        batch_poses = []
        sequence_len = self.sequence_len
        for _ in range(batch_size):
            
            input_images, velocities, poses = self.get_series(sequence_len)
            batch_input.append(input_images)
            batch_velocities.append(velocities)
            batch_poses.append(poses)

        return np.stack(batch_input), np.stack(velocities), np.stack(poses)

    def get_series(self, sequences = None, ):
        if sequences is None:
           sequences = self.sequences
        self.seq_idx %= len(self.sequences)
        while self.sequences[self.seq_idx] not in sequences:
            self.seq_idx += 1
        sequence_len = self.sequence_len
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

    # new functions to load data
    def get_series_batch_train(self, batch_size = 10,sequences = None, ):
        batch_input = []
        batch_velocities = []
        batch_poses = []
        batch_initial_poses = []
        sequence_len = self.sequence_len
        for _ in range(batch_size):
            
            input_images, velocities, poses, initial_pose = self.get_series_train()
            batch_input.append(input_images)
            batch_velocities.append(velocities)
            batch_poses.append(poses)
            batch_initial_poses.append(initial_pose)

        return np.stack(batch_input), np.stack(batch_velocities), np.stack(batch_poses), np.stack(batch_initial_poses)

    #loads all the possible sequences in the training set all at once
    def load_data_train(self,sequences = None,):
        train_input = []
        train_velocities = []
        train_poses = []
        sequence_len = self.sequence_len
        if sequences is None:
           sequences = self.sequences
        for sequence in sequences:
            train_mask = self.train_mask[sequence]
            train_mask_len = len(train_mask)
            train_idx = self.train_idx[sequence]
            
            
            while(train_idx<train_mask_len):
                series_input = self.input[sequence][train_mask[train_idx]:train_mask[train_idx]+sequence_len]
                velocities = self.velocities[sequence][train_mask[train_idx]:train_mask[train_idx]+sequence_len]
                poses = self.poses[sequence][train_mask[train_idx]:train_mask[train_idx]+sequence_len]
                train_idx+=1
                print(train_idx)
                train_input.append(series_input)
                train_velocities.append(velocities)
                train_poses.append(poses)
            
        return np.stack(train_input),np.stack(train_velocities),np.stack(train_poses)

    def load_data_validation(self, sequences = None,):
        val_input = []
        val_velocities = []
        val_poses = []
        sequence_len = self.sequence_len
        if sequences is None:
           sequences = self.sequences
        for sequence in sequences:
            val_mask = self.val_mask[sequence]
            val_mask_len = len(val_mask)
            val_idx = self.val_idx[sequence]
            
            
            while(val_idx<val_mask_len):
                series_input = self.input[sequence][val_mask[val_idx]:val_mask[val_idx]+sequence_len]
                velocities = self.velocities[sequence][val_mask[val_idx]:val_mask[val_idx]+sequence_len]
                poses = self.poses[sequence][val_mask[val_idx]:val_mask[val_idx]+sequence_len]
                val_idx+=1
                val_input.append(series_input)
                val_velocities.append(velocities)
                val_poses.append(poses)
            
        return np.stack(val_input),np.stack(val_velocities),np.stack(val_poses)

    def load_data_test(self, sequences = None,):
        test_input = []
        test_velocities = []
        test_poses = []
        sequence_len = self.sequence_len
        if sequences is None:
           sequences = self.sequences
        for sequence in sequences:
            test_mask = self.test_mask[sequence]
            test_mask_len = len(test_mask)
            test_idx = self.test_idx[sequence]
            
            
            while(test_idx<test_mask_len):
                series_input = self.input[sequence][test_mask[test_idx]:test_mask[test_idx]+sequence_len]
                velocities = self.velocities[sequence][test_mask[test_idx]:test_mask[test_idx]+sequence_len]
                poses = self.poses[sequence][test_mask[test_idx]:test_mask[test_idx]+sequence_len]
                test_idx+=1
                test_input.append(series_input)
                test_velocities.append(velocities)
                test_poses.append(poses)
            
        return np.stack(test_input),np.stack(test_velocities),np.stack(test_poses)
    
    def get_series_train(self, sequences = None,):
        if sequences is None:
           sequences = self.sequences
        sequence_len = self.sequence_len
        seq_num = np.random.randint(0,len(sequences))
        selected_sequence = sequences[seq_num]
        train_mask = self.train_mask[selected_sequence]

        train_mask_len = len(train_mask)
        train_idx = self.train_idx[selected_sequence]
            
        series_input = self.input[selected_sequence][train_mask[train_idx]:train_mask[train_idx]+sequence_len]
        velocities = self.velocities[selected_sequence][train_mask[train_idx+1]:train_mask[train_idx+1]+sequence_len]
        poses = self.poses[selected_sequence][train_mask[train_idx+1]:train_mask[train_idx+1]+sequence_len]
        initial_pose = self.poses[selected_sequence][train_mask[train_idx]]
        self.train_idx[selected_sequence]+=1

        if(self.train_idx[selected_sequence]>=train_mask_len-1):
            self.train_idx[selected_sequence] = 0
            #print('training set gone through')
           
            
        return series_input, velocities, poses, initial_pose
        
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



































import os
import numpy as np
import cv2
import sys
#sys.path.append('F:\\ComputerVision\\VisualOdometry')

from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm

#os.chdir('F:\\ComputerVision\\VisualOdometry')

class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images = self._load_images(os.path.join(data_dir,'image_l'))
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    @staticmethod
    def _load_calib(file_path):
        with open (file_path ,'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep =' ')
            P = params.reshape(3,4)
            K = P[0:3, 0:3]

        return K, P
    
    @staticmethod
    def _load_poses(file_path):
        poses = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                params = np.fromstring(line, dtype=np.float64, sep=' ')
                T = params.reshape(3,4)
                T = np.vstack((T, [0,0,0,1]))
                poses.append(T)

        return poses
        
    @staticmethod
    def _load_images(file_path):
        image_paths = [os.path.join(file_path, file) for file in sorted(os.listdir(file_path))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
                
    @staticmethod
    def _form_transf(R,t):
        # T = np.hstack((R,t.reshape(-1,1)))
        # T = np.vstack((T,[0,0,0,1]))
        # return T
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T        

    def get_matches(self, i):

        #Finding keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(self.images[i-1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i], None)

        #Matching descriptors
        matches = self.flann.knnMatch(des1, des2, k=2)

        #only good match
        #matches_mask = [[0,0] for i in range(len(matches))]

        good = []
        try:
            for m, n in matches:
            #for i, (m, n) in enumerate(matches):
                if m.distance < 0.8 *n.distance:
                    #matches_mask[i] = [1, 0]
                    good.append(m)
        except ValueError:
            pass

        # draw_params = dict(matchColor=(0, 255, 0),  # Matched features in green color
        #                 singlePointColor=(0, 0, 255),  # Unmatched features in red color
        #                 matchesMask= matches_mask,
        #                 flags=0)
        draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)
        # draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
        #          singlePointColor = (0, 0, 255),
        #          matchesMask = None, # draw only inliers
        #          flags = 2)
                        
        img_matches = cv2.drawMatches(self.images[i], kp1, self.images[i-1], kp2, good, None, **draw_params)
        cv2.imshow("image",img_matches)
        cv2.waitKey(200)

        #Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        return q1,q2

    def get_pose(self, q1, q2):

        # Essential Matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold = 1)

        #Decompose the Essential Matrix into R, t
        R, t = self.decomp_essential_mat(E, q1, q2)

        #Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix
    
    def decomp_essential_mat(self, E, q1, q2):

        def sum_z_cal_relative_scale(R, t):
            # Transformation matiix
            T = self._form_transf(R, t)
            # Projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3,1))), axis=1), T)

            #Triangulate 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            #from cam2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un_homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3,:]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3,:]

            # Find the number of points there has positive z coordinates in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2,:] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2,:] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale 
        
        # Decompose essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        #Make a list of the different possible paors
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        #Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinates
            right_pair_idx = np.argmax(z_sums)
            right_pair = pairs[right_pair_idx]
            relative_scale = relative_scales[right_pair_idx]
            R1, t = right_pair
            t = t * relative_scale

            return [R1, t]





def main():
    data_dir = 'KITTI_sequence_2'
    vo = VisualOdometry(data_dir)

    play_trip(vo.images)

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1,q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose,  np.linalg.inv(transf))
            gt_path.append((gt_pose[0,3], gt_pose[2,3]))
            estimated_path.append((cur_pose[0,3], cur_pose[2,3]))
    plotting.visualize_paths(gt_path,estimated_path, "Visual Odometery", file_out=os.path.basename(data_dir)+ ".html")

if __name__ == "__main__":
    main()
import glob
import os
import numpy as np
import cv2
import dlib
from scipy.spatial import distance
import time

def landmarks_to_np(landmarks, dtype="int", normalize=True):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    if normalize:
        left = np.mean(coords[36:42], axis=0).astype(np.int32)
        right = np.mean(coords[42:48], axis=0).astype(np.int32)
        d = distance.euclidean(left, right)
        m = (left + right) / 2        
        coords = (coords - m) / d
    return coords

def detect_landmarks(image, detector, predictor):
    recs = detector(image, 1)
    landmarks = []
    if recs[0]:
        landmarks = landmarks_to_np(predictor(image, recs[0]))

    t, b, l, r = recs[0].top(), recs[0].bottom(), recs[0].left(), recs[0].right()
    image = image[t:b, l:r, :]

    #if len(landmarks):
    #    landmarks -= [l, t]

    return image, landmarks

def get_labels(path, n_labels=65):
    labels = np.zeros(n_labels)
    facs = np.loadtxt(path)
    if len(facs.shape) < 2:
        facs = np.expand_dims(facs, axis=0)
    for fac in facs:
        labels[int(fac[0])] = 1
    return labels

def main():
    root = '/home/ubuntu/CourseAI/datasets/facs/'
    X, Y = [], []
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    n_persons = len(os.listdir(root + 'images/'))
    
    t0 = time.time()
    
    for i, person_id in enumerate(os.listdir(root + 'images/')):
        for clip_id in os.listdir(root + 'images/' + person_id):
            x_imgs, x_lands = [], []

            clip_path = root + 'images/' + person_id + '/' + clip_id + '/'

            lbl_paths = glob.glob(clip_path.replace('images', 'labels') + '*.txt')
            if len(lbl_paths):
                labels = get_labels(lbl_paths[0])

                for img_path in sorted(glob.glob(clip_path + '*.png')):
                    image, landmarks = detect_landmarks(cv2.imread(img_path), detector, predictor)
                    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                    x_imgs.append(image)
                    x_lands.append(landmarks)
                
                X.append([x_imgs, x_lands])
                Y.append(np.array(labels)) # possible ERROR
        
        print(i+1, 'persons out of', n_persons)
        
    np.save(root + 'X.npy', X)
    np.save(root + 'Y.npy', Y)
    
    
    print('Preparation is done. s', time.time() - t0)
    return 0

if __name__ == '__main__':
    main()
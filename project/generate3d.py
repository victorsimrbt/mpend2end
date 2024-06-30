import cv2
import os
import blur_detector
from matplotlib import pyplot as plt
imgs = [img for img in os.listdir(r"../captures")][1:]
cnt = 0
blurs = []
images = []
for img in imgs:
    path = os.path.join(r"../captures",img)
    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    images.append(image)
    print(image.shape)
    print("image",cnt)
    blur_map = blur_detector.detectBlur(image, downsampling_factor=1, show_progress = False, num_scales=4, scale_start=2, num_iterations_RF_filter=3)
    blurs.append(blur_map)
    plt.imshow(image,cmap="grey")
    plt.imshow(blurs[cnt],alpha = 0.6)
    cnt += 1
    plt.show()
    
import cv2
import numpy as np

def align_images_sift(img1, img2, idx):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors using SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # ! scale invariant feature transform

    FLANN_INDEX_KDTREE = 1
    # ! optimised form of nearest neighbors
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    MIN_MATCH_COUNT = 10
    if len(good_matches) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # ! find destination points for each point in source
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # ! estimate homography between both sets
        aligned_img = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
        aligned_img = cv2.warpPerspective(blurs[idx], H, (img2.shape[1], img2.shape[0]))

        return aligned_img, H

    else:
        raise ValueError("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))

base = image[0]
contour = np.zeros((480,640))
depths = [0,0,0.13,0.01,0.01,0.1,0.07,0,0,0,0.07,0.03,0.01,0.02]
# ! determine from camera intrinsics for our data
depths = np.array(depths)*1.2
for i in range(len(images)):
    if i == 2 or i == len(images)-1:
        continue
    try:
        aligned_img, H = align_images_sift(images[i],images[0],i)
    except:
        # ! sift alignment may fail in some cases
        pass
    
    print(depths[i],aligned_img)
    min_val = np.min(aligned_img)
    max_val = np.max(aligned_img)

    aligned_img = 0.05 * (aligned_img - min_val) / (max_val - min_val)
    contour = np.maximum(contour,depths[i]*(aligned_img>0.01)+aligned_img*3*(aligned_img>0.01))
    
    plt.imshow(depths[i]+aligned_img/10)
    print(np.max(depths[i]+aligned_img/10))
    plt.show()


def extract_contour(contour):
    xs = []
    ys = []
    zs = []
    cs = []
    thresh = 0.01
    for i in range(contour.shape[0]):
        for j in range(contour.shape[1]):
            if contour[i][j] > thresh:
                # ! only record points that have sufficiently high height
                xs.append(i)
                ys.append(j)
                zs.append(contour[i][j])
                # print(image[0][i][j])
                cs.append(images[0][i][j])
    plt.imshow(contour)
    # ! return color, position and height for reconstruction
    return cs,xs,ys,zs

cs,xs,ys,zs = extract_contour(contour)
save = np.load("point_cloud.npy")
print(len(xs))
#!/usr/bin/env python
# coding: utf-8

# In[14]:


def process_image(original_img,img,img_new,img_after_adaptive_non_max_suppression,img_after_rotational_invariance):
    height = original_img.shape[0]
    width = original_img.shape[1]
    c_array,threshold,magnitude,orientation,img = calculate_c_array(original_img,height,width,img)
    keypoints,keypoint_corner_response_list,img_new = generate_keypoints(c_array,threshold,height,width,img_new)
    keypoints,img_after_adaptive_non_max_suppression = adaptive_non_maximum_suppression(keypoint_corner_response_list,img_after_adaptive_non_max_suppression)
    keypoints,magnitude,orientation = assignOrientation(magnitude,orientation,keypoints)
    img_descriptors,keypoints,img_after_rotational_invariance = create_image_descriptors(magnitude,orientation,keypoints,height,width,img_after_rotational_invariance)
    return img_descriptors,keypoints,img,img_new,img_after_adaptive_non_max_suppression,img_after_rotational_invariance


# ## Feature Detection

# In[15]:


def paintKeypoints(img,y,x):
    img.itemset((y-1,x-1,2),255)
    img.itemset((y-1,x,2),255)
    img.itemset((y-1,x+1,2),255)
    img.itemset((y,x-1,2),255)
    img.itemset((y,x+1,2),255)
    img.itemset((y+1,x-1,2),255)
    img.itemset((y+1,x,2),255)
    img.itemset((y+1,x+1,2),255)
    return img


# In[25]:


def calculate_c_array(original_img,height,width,img):

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    print(gray_img.shape)
    # 1st order derivative w.r.t x and y
    dx = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=5) / 8
    dy = cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=5) / 8

    I_x2 = dx ** 2
    I_y2 = dy ** 2
    I_xy = dx * dy

    w_size = 5
    offset = int(w_size / 2)
    
    # creating a 5 x 5 gaussian kernel
    gaussian_kernel = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]]) / 273
    # creating a numpy array to store response values calculated using Harris matrix
    c_array = np.zeros((height,width))
    maximum = 0
    for y in range(offset,height-offset):
        for x in range(offset,width-offset):

            # selecting window pixels from all three double differentiation numpy arrays
            W_I_x2 = I_x2[y - offset:y + offset + 1,x-offset:x+offset+1] * gaussian_kernel
            W_I_y2 = I_y2[y - offset:y + offset + 1,x-offset:x+offset+1] * gaussian_kernel
            W_I_xy = I_xy[y - offset:y + offset + 1,x-offset:x+offset+1] * gaussian_kernel

            W_I_x2_sum = W_I_x2.sum()
            W_I_y2_sum = W_I_y2.sum()
            W_I_xy_sum = W_I_xy.sum()

            det = (W_I_x2_sum * W_I_y2_sum) - (W_I_xy_sum ** 2)
            trace = W_I_x2_sum + W_I_y2_sum

            c = det / trace
            
            c_array[y,x] = c
            if c > maximum:
                maximum = c
    # .45
    threshold = 0.4 * maximum
    
    for y in range(offset,height-offset):
        for x in range(offset,width-offset):
            c = c_array[y,x]

            if (c > threshold):
                img = paintKeypoints(img,y,x)
    
    # calculate magnitude of intensity
    magnitude = np.sqrt(np.square(dx)+np.square(dy))
    # calculate orientation of each pixel around the keypoint
    orientation = np.degrees(np.arctan(dy/dx))
    orientation = np.where(orientation<0, orientation+360, orientation) # convert negative angles to positive by adding 360
    return c_array,threshold,magnitude,orientation,img


# In[17]:


# select interest points which are local maximum in a 3x3 neighborhood
def generate_keypoints(c_array,threshold,height,width,img_new):
    offset = 1
    keypoint_corner_response_list = []
    
    for y in range(offset,height-offset):
        for x in range(offset,width-offset):

            # selecting window pixels from c_arraay
            W_values = c_array[y - offset:y + offset + 1,x-offset:x+offset+1]
            maximum = np.amax(W_values)
            c = c_array[y,x]

            if (c > threshold) and (c==maximum):
                img_new = paintKeypoints(img_new,y,x)
                keypoint_corner_response_list.append([y,x,c])
            else:
                c_array[y,x]=0
    keypoints = np.transpose(np.nonzero(c_array))
    return keypoints,keypoint_corner_response_list,img_new


# In[18]:


from operator import itemgetter
import math
# select interest point using adaptive non maximum suppression
def adaptive_non_maximum_suppression(keypoint_corner_response_list,img_after_adaptive_non_max_suppression):
    new_keypoints = np.empty( shape=(0, 2) )
    
    keypoint_corner_response_list = sorted(keypoint_corner_response_list,key=itemgetter(2),reverse=True)
    keypoint_corner_response_radius_list = []
    for idx,entry in enumerate(keypoint_corner_response_list):
        
        if idx==0:
            keypoint_corner_response_radius_list.append([entry[0],entry[1],entry[2],0])
        else:
            
            id_temp = idx
            smallest_distance = sys.maxsize
            while id_temp!=0:
                id_temp = id_temp - 1
                y_dist = math.pow((keypoint_corner_response_list[id_temp][0] - keypoint_corner_response_list[idx][0]),2)
                x_dist = math.pow((keypoint_corner_response_list[id_temp][1] - keypoint_corner_response_list[idx][1]),2)
                distance = math.sqrt(y_dist+x_dist)
                if distance < smallest_distance:
                    smallest_distance = distance
            keypoint_corner_response_radius_list.append([entry[0],entry[1],entry[2],smallest_distance])
    first_key_point = [keypoint_corner_response_radius_list[0][0],keypoint_corner_response_radius_list[0][1]]
    new_keypoints = np.append(new_keypoints,np.array(first_key_point).reshape(-1,2),axis=0)
    keypoint_corner_response_radius_list.pop(0)
    keypoint_corner_response_radius_list = sorted(keypoint_corner_response_radius_list,key=itemgetter(3),reverse=True)
    keypoint_corner_response_radius_list = np.array(keypoint_corner_response_radius_list)
    range_of_keypoints = int(0.6 * len(keypoint_corner_response_radius_list))
    rem_keypoints = keypoint_corner_response_radius_list[:range_of_keypoints,[0,1]]
    new_keypoints = np.append(new_keypoints,np.array(rem_keypoints).reshape(-1,2),axis=0)
    new_keypoints = new_keypoints.astype(int)
    for y,x in new_keypoints:
        img_after_adaptive_non_max_suppression = paintKeypoints(img_after_adaptive_non_max_suppression,y,x)
    return new_keypoints,img_after_adaptive_non_max_suppression


# ## Feature Description

# In[19]:


def getIndicesWithMoreThanEightyPercent(arr):
    maximum = np.max(arr)
    indices = np.where((arr>(0.8*maximum)) & (arr<maximum))
    return indices[0]

def calculate_new_keypoint_indices(new_ind,y,x):
    new_y = 0
    new_x = 0
    temp_x = new_ind[0]
    temp_y = new_ind[1]
    
    if temp_x == 0:
        new_y = y - 1
    elif temp_x == 1:
        new_y = y
    else:
        new_y = y + 1
        
    if temp_y == 0:
        new_x = x - 1
    elif temp_y == 1:
        new_x = x
    else:
        new_x = x + 1
    return new_y,new_x


# ### Rotational Invariance

# In[20]:


# Assign orientation for rotational invariance
def assignOrientation(magnitude,orientation,keypoints):
    offset = 1
    new_keypoints = np.empty( shape=(0, 2) )
    
    for kp in keypoints:
        y = kp[0]
        x = kp[1]
        temp_magnitude = magnitude[y - offset:y + offset + 1,x-offset:x+offset+1]
        temp_orientation = orientation[y - offset:y + offset + 1,x-offset:x+offset+1]
        
        # creted a histogram with orientation bins and magnitude contributing to corresponding bins
        his = np.histogram(temp_orientation,range=(0,360), bins=36, weights=temp_magnitude)
        
        # calculating new orientation of keypoint
        new_orientation = his[1][np.argmax(his[0])] + 5
        
        # setting it
        orientation[y - offset:y + offset + 1,x-offset:x+offset+1]=new_orientation
        
        # checking if any bin has value more than 80% of the largest bin
        indices = getIndicesWithMoreThanEightyPercent(his[0])
        if len(indices) > 0:
            for i in indices:
                ind = list(np.where((temp_orientation>(i*10)) & (temp_orientation<(i*10+10))))
                if len(ind[0]) > 0:
                    maximum = np.amax(temp_magnitude[ind])
                    new_ind = np.transpose(ind)[np.where(temp_magnitude[ind]==maximum)][0]
                    new_keypoint = calculate_new_keypoint_indices(new_ind,y,x)

                    # setting magnitude and orientation of the new key_point
                    magnitude[new_keypoint[0],new_keypoint[1]] = magnitude[y,x]
                    orientation[new_keypoint[0],new_keypoint[1]] = (i*10) + 5

                    new_keypoints = np.append(new_keypoints,np.array(new_keypoint).reshape(-1,2),axis=0)
    keypoints = np.append(keypoints,new_keypoints,axis=0)
    keypoints = np.unique(keypoints,axis=0)
    return keypoints,magnitude,orientation


# ### Generate 128 bit feature descriptors

# In[29]:


def normalize_and_threshold_descriptor(descriptor):
    
    descriptor_norm = descriptor / np.sqrt(np.sum(np.square(descriptor)))
    np.place(descriptor_norm,descriptor_norm>0.19,0.19)
    new_descriptor_norm = descriptor_norm / np.sqrt(np.sum(np.square(descriptor_norm)))
    return new_descriptor_norm

def create_image_descriptors(magnitude,orientation,keypoints,height,width,img_after_rotational_invariance):
    offset = 8
    img_descriptors = np.empty( shape=(0, 128) )
    new_keypoints = np.empty( shape=(0, 2) )
    for kp in keypoints:
        y = int(kp[0])
        x = int(kp[1])

        if (y in range(offset,height-offset)) & (x in range(offset,width-offset)):
            descriptor = np.empty((0,0))
            magnitude_16 = magnitude[int(y - offset):int(y + offset),int(x - offset):int(x + offset)]
            orientation_16 = orientation[int(y - offset):int(y + offset),int(x - offset):int(x + offset)]

            magnitude_4 = magnitude_16.reshape(16//4, 4, -1, 4).swapaxes(1,2).reshape(-1, 4, 4)
            orientation_4 = orientation_16.reshape(16//4, 4, -1, 4).swapaxes(1,2).reshape(-1, 4, 4)

            for i in range(0,16):
                # created a histogram with orientation bins and magnitude contributing to corresponding bins
                temp_magnitude = magnitude_4[i]
                temp_orientation = orientation_4[i]
                his = np.histogram(temp_orientation,range=(0,360), bins=8, weights=temp_magnitude)
                descriptor = np.append(descriptor,his[0])
            descriptor = np.array(descriptor).reshape(-1,128)
            normalized_descriptor = normalize_and_threshold_descriptor(descriptor)
            img_descriptors = np.append(img_descriptors,normalized_descriptor,axis=0)
            new_keypoints = np.append(new_keypoints,np.array(kp).reshape(-1,2),axis=0)
            img_after_rotational_invariance = paintKeypoints(img_after_rotational_invariance,y,x)
               
        
    return img_descriptors,new_keypoints,img_after_rotational_invariance
 


# ## Feature Matching

# In[22]:


import sys
def isDistanceRatioValid(distance_arr,smallest_distance):
    matching_feature_index = distance_arr.index(smallest_distance)
    distance_arr.sort() 
    second_smallest = distance_arr[1]
    ratio = smallest_distance / second_smallest
    if ratio > 0.8:
        return False,matching_feature_index
    else:
        return True,matching_feature_index

def matchFeatures(img_descriptor_1,img_descriptor_2,keypoints_1,keypoints_2):
    matching_keypoints_1 =[]
    matching_keypoints_2 =[]
    distances = []
    smallest_distance_of_all = sys.maxsize
    for idx,des1 in enumerate(img_descriptor_1):
        distance_arr = []
        for des2 in img_descriptor_2:
            distance = 0
            for i in range(0,128):
                distance = distance + abs((des1[i]*des1[i]) - (des2[i]*des2[i]))
            distance_arr.append(distance)
        smallest_distance = min(distance_arr)
        
        ret_values = isDistanceRatioValid(distance_arr,smallest_distance)
        if ret_values[0]:
            matching_keypoints_1.append(keypoints_1[idx])
            matching_keypoints_2.append(keypoints_2[ret_values[1]])
            distances.append(smallest_distance)
            if smallest_distance < smallest_distance_of_all:
                smallest_distance_of_all = smallest_distance
    threshold = 1.8 * smallest_distance_of_all
    for idx,distance in enumerate(distances):
        if distance > threshold:
            matching_keypoints_1.pop(idx)
            matching_keypoints_2.pop(idx)
            distances.pop(idx)
    return matching_keypoints_1,matching_keypoints_2


# In[23]:


def create_keypoints_matching_image(img_after_rotational_invariance_1, img_after_rotational_invariance_2,matching_keypoints_1,matching_keypoints_2):
    match_image = np.concatenate((img_after_rotational_invariance_1, img_after_rotational_invariance_2), axis=1)
    color = (0, 255, 0) 
    thickness = 1
    for kp1,kp2 in zip(matching_keypoints_1,matching_keypoints_2):
        width = img_after_rotational_invariance_1.shape[1]
        kp_1 = (int(kp1[1]),int(kp1[0]))
        kp_2 = (int(kp2[1]+width),int(kp2[0]))
        numpy_horizontal_concat = cv2.line(match_image, kp_1, kp_2,color,thickness)
    return match_image

def improve_matches(matching_keypoints_11,matching_keypoints_21,matching_keypoints_12,matching_keypoints_22):
    matching_keypoints_1 = np.concatenate((np.array(matching_keypoints_11), np.array(matching_keypoints_21)), axis=1)
    matching_keypoints_2 = np.concatenate((np.array(matching_keypoints_12), np.array(matching_keypoints_22)), axis=1)
    r, c = np.array(matching_keypoints_1).shape
    dtype={'names':['f{}'.format(i) for i in range(c)],
           'formats':c * [matching_keypoints_1.dtype]}
    common_points = np.intersect1d(matching_keypoints_1.view(dtype), matching_keypoints_2.view(dtype))
    print(common_points.shape)
    print(common_points)
    new_keypoints_1 = np.empty( shape=(0, 2) )
    new_keypoints_2 = np.empty( shape=(0, 2) )
    for i in common_points:
        arr_1 = np.array([[i[0],i[1]]]).reshape(-1,2)
        new_keypoints_1 = np.append(new_keypoints_1,arr_1,axis=0)
        arr_2 = np.array([[i[2],i[3]]]).reshape(-1,2)
        new_keypoints_2 = np.append(new_keypoints_2,arr_2,axis=0)
    print(new_keypoints_1)
    print(new_keypoints_2)
    return new_keypoints_1,new_keypoints_2
# In[38]:


import cv2
import numpy as np
# yosemite\Yosemite1.jpg
# panorama\pano1_0009.jpg
# graf\img2.ppm
img_name_1 = input('Enter path of image 1 : ')
original_img = cv2.imread('image_sets\\'+img_name_1)
img_1 = original_img.copy()
img_new_1 = original_img.copy()
img_after_adaptive_non_max_suppression_1 = original_img.copy()
img_after_rotational_invariance_1 = original_img.copy()
better_matches_1 = original_img.copy()
image_descriptor_1,keypoints_1,img_1,img_new_1,img_after_adaptive_non_max_suppression_1,img_after_rotational_invariance_1 = process_image(original_img,img_1,img_new_1,img_after_adaptive_non_max_suppression_1,img_after_rotational_invariance_1)
print(keypoints_1.shape)
print('1st done')
# yosemite\Yosemite2.jpg
# panorama\pano1_0010.jpg
# graf\img4.ppm
img_name_2 = input('Enter path of image 2 : ')
original_img = cv2.imread('image_sets\\'+img_name_2)
img_2 = original_img.copy()
img_new_2 = original_img.copy()
img_after_adaptive_non_max_suppression_2 = original_img.copy()
img_after_rotational_invariance_2 = original_img.copy()
better_matches_2 = original_img.copy()
image_descriptor_2,keypoints_2,img_2,img_new_2,img_after_adaptive_non_max_suppression_2,img_after_rotational_invariance_2 = process_image(original_img,img_2,img_new_2,img_after_adaptive_non_max_suppression_2,img_after_rotational_invariance_2)
print(keypoints_2.shape)

matching_keypoints_11,matching_keypoints_21 = matchFeatures(image_descriptor_1,image_descriptor_2,keypoints_1,keypoints_2)
print('matching_keypoints_11')
print(matching_keypoints_11)
print('matching_keypoints_21')
print(matching_keypoints_21)
match_image_before = create_keypoints_matching_image(img_after_rotational_invariance_1, img_after_rotational_invariance_2,matching_keypoints_11,matching_keypoints_21)
matching_keypoints_22,matching_keypoints_12 = matchFeatures(image_descriptor_2,image_descriptor_1,keypoints_2,keypoints_1)
print('matching_keypoints_12')
print(matching_keypoints_12)
print('matching_keypoints_22')
print(matching_keypoints_22)

new_keypoints_1,new_keypoints_2 = improve_matches(matching_keypoints_11,matching_keypoints_21,matching_keypoints_12,matching_keypoints_22)
match_image_after = create_keypoints_matching_image(better_matches_1, better_matches_2,new_keypoints_1,new_keypoints_2)

cv2.imshow('Corners detected using threshold value_1',img_1)
cv2.imshow('Corners detected using threshold value and 3x3 local maxima_1',img_new_1)
cv2.imshow('img_after_rotational_invariance_1',img_after_rotational_invariance_1)
cv2.imshow('img_after_adaptive_non_max_suppression_1',img_after_rotational_invariance_1)
cv2.imshow('Corners detected using threshold value_2',img_2)
cv2.imshow('Corners detected using threshold value and 3x3 local maxima_2',img_new_2)
cv2.imshow('img_after_adaptive_non_max_suppression_2',img_after_rotational_invariance_2)
cv2.imshow('img_after_rotational_invariance_2',img_after_rotational_invariance_2)
cv2.imshow('match_image',match_image_before)
cv2.imshow('match_image_after',match_image_after)
cv2.waitKey()


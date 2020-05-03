#!/usr/bin/env python
# coding: utf-8

# In[26]:


import cv2
import os
import numpy as np
import sys
from random import randint
import math


# In[27]:


def paintKeypoints(img,y,x):
    radius = 5
    color = (0, 0, 255) 
    thickness = 1
    img = cv2.circle(img, (x,y), radius, color, thickness) 
    return img


# In[28]:


def calculate_c_array(original_img,height,width):

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
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
    threshold = 0.3 * maximum
    
    # calculate magnitude of intensity
    magnitude = np.sqrt(np.square(dx)+np.square(dy))
    # calculate orientation of each pixel around the keypoint
    orientation = np.degrees(np.arctan(dy/dx))
    orientation = np.where(orientation<0, orientation+360, orientation) # convert negative angles to positive by adding 360
    return c_array,threshold,magnitude,orientation


# In[29]:


# select interest points which are local maximum in a 3x3 neighborhood
def generate_keypoints(c_array,threshold,height,width):
    offset = 1
    keypoint_corner_response_list = []
    
    for y in range(offset,height-offset):
        for x in range(offset,width-offset):

            # selecting window pixels from c_arraay
            W_values = c_array[y - offset:y + offset + 1,x-offset:x+offset+1]
            maximum = np.amax(W_values)
            c = c_array[y,x]

            if (c > threshold) and (c==maximum):
                keypoint_corner_response_list.append([y,x,c])
            else:
                c_array[y,x]=0
    keypoints = np.transpose(np.nonzero(c_array))
    return keypoints,keypoint_corner_response_list


# In[30]:


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
    range_of_keypoints = int(0.4 * len(keypoint_corner_response_radius_list))
    rem_keypoints = keypoint_corner_response_radius_list[:range_of_keypoints,[0,1]]
    new_keypoints = np.append(new_keypoints,np.array(rem_keypoints).reshape(-1,2),axis=0)
    new_keypoints = new_keypoints.astype(int)
    for y,x in new_keypoints:
        img_after_adaptive_non_max_suppression = paintKeypoints(img_after_adaptive_non_max_suppression,y,x)
    return new_keypoints,img_after_adaptive_non_max_suppression


# In[31]:


def compute_and_paint_harris_keypoints(original_img,return_image_name):
    height = original_img.shape[0]
    width = original_img.shape[1]
    c_array,threshold,magnitude,orientation = calculate_c_array(original_img,height,width)
    keypoints,keypoint_corner_response_list = generate_keypoints(c_array,threshold,height,width)
    keypoints,img_after_adaptive_non_max_suppression = adaptive_non_maximum_suppression(keypoint_corner_response_list,original_img)
    cv2.imwrite(return_image_name,img_after_adaptive_non_max_suppression)
    return keypoints,magnitude,orientation,height,width


# In[72]:


original_img = cv2.imread('project_images\Boxes.png')
compute_and_paint_harris_keypoints(original_img,'1a.png')
print('1a.png has been stored in current directory depicting corners in image Boxes.png')

original_img_1 = cv2.imread('project_images\Rainier_2_images\Rainier1.png')
img_1 = original_img_1.copy()
keypoints_1,magnitude_1,orientation_1,height_1,width_1 = compute_and_paint_harris_keypoints(original_img_1,'1b.png')
print('1b.png has been stored in current directory depicting corners in image Rainier1.png')

original_img_2 = cv2.imread('project_images\Rainier_2_images\Rainier2.png')
img_2 = original_img_2.copy()
keypoints_2,magnitude_2,orientation_2,height_2,width_2 = compute_and_paint_harris_keypoints(original_img_2,'1c.png')
print('1c.png has been stored in current directory depicting corners in image Rainier2.png')


# # Step 2:

# # Feature description

# In[33]:


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

def normalize_and_threshold_descriptor(descriptor):
    
    descriptor_norm = descriptor / np.sqrt(np.sum(np.square(descriptor)))
    np.place(descriptor_norm,descriptor_norm>0.19,0.19)
    new_descriptor_norm = descriptor_norm / np.sqrt(np.sum(np.square(descriptor_norm)))
    return new_descriptor_norm

def create_image_descriptors(magnitude,orientation,keypoints,height,width):
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
    return img_descriptors,new_keypoints


# In[35]:


import sys
def isDistanceRatioValid_1(distance_arr,smallest_distance):
    matching_feature_index = distance_arr.index(smallest_distance)
    distance_arr.sort() 
    second_smallest = distance_arr[1]
    ratio = smallest_distance / second_smallest
    if ratio > 0.8:
        return False,matching_feature_index
    else:
        return True,matching_feature_index

def matchFeatures_1(img_descriptor_1,img_descriptor_2,keypoints_1,keypoints_2):
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
        
        ret_values = isDistanceRatioValid_1(distance_arr,smallest_distance)
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


# In[63]:


def create_keypoints_matching_image(img_1, img_2,keypoints_1,keypoints_2,return_image_name,option):
    height1 = img_1.shape[0]
    height2 = img_2.shape[0]
    width1 = img_1.shape[1]
    width2 = img_2.shape[1]
    img2_new = img_2
    img1_new = img_1
    if height1!=height2:
        if height1>height2:
            img2_new = np.zeros( shape=(height1, width2,3) )
            img2_new[:height2,:width2] = img_2
        else:
            img1_new = np.zeros( shape=(height2, width1,3) )
            img1_new[:height1,:width1] = img_1
    
    numpy_horizontal_concat = np.concatenate((img1_new, img2_new), axis=1)
    color = (0, 255, 0) 
    thickness = 1
    for kp1,kp2 in zip(keypoints_1,keypoints_2):
        width = img_1.shape[1]
        if option==1:
            kp_1 = (int(kp1[0]),int(kp1[1]))
            kp_2 = (int(kp2[0]+width),int(kp2[1]))
        else:
            kp_1 = (int(kp1[1]),int(kp1[0]))
            kp_2 = (int(kp2[1]+width),int(kp2[0]))
        numpy_horizontal_concat = cv2.line(numpy_horizontal_concat, kp_1, kp_2,color,thickness)
    cv2.imwrite(return_image_name,numpy_horizontal_concat)


# In[73]:


keypoints_1,magnitude_1,orientation_1 = assignOrientation(magnitude_1,orientation_1,keypoints_1)
img_descriptor_1,keypoints_1 = create_image_descriptors(magnitude_1,orientation_1,keypoints_1,img_1.shape[0],img_1.shape[1])
keypoints_2,magnitude_2,orientation_2 = assignOrientation(magnitude_2,orientation_2,keypoints_2)
img_descriptor_2,keypoints_2 = create_image_descriptors(magnitude_2,orientation_2,keypoints_2,img_2.shape[0],img_2.shape[1])
keypoints_1,keypoints_2 = matchFeatures_1(img_descriptor_1,img_descriptor_2,keypoints_1,keypoints_2)
create_keypoints_matching_image(img_1, img_2,keypoints_1,keypoints_2,'2.png',0)
print('\n')
print('2.png has been stored in current directory depicting matching interest points between Rainier1.png and Rainier2.png')


# ## Feature Matching

# In[42]:


def isDistanceRatioValid(distance_arr):
    distance_arr = sorted(distance_arr,key = lambda element : element[1])
    smallest_distance = distance_arr[0][1]
    second_smallest = distance_arr[1][1]
    ratio = smallest_distance / second_smallest
    if ratio > 0.8:
        return False,distance_arr[0][0],smallest_distance
    else:
        return True,distance_arr[0][0],smallest_distance


# In[43]:


def matchFeatures(img_descriptor_1,img_descriptor_2,keypoints_1,keypoints_2,matches):
    matching_keypoints_1 =[]
    matching_keypoints_2 =[]
    updated_matching_keypoints_1 =[]
    updated_matching_keypoints_2 =[]
    distances = []
    smallest_distance_of_all = sys.maxsize
    for m in matches:
        kp1_idx = m.queryIdx
        des1 = img_descriptor_1[kp1_idx]
        distance_arr = []
        for m in matches:
            kp2_idx = m.trainIdx
            des2 = img_descriptor_2[kp2_idx]
            distance = 0
            for i in range(0,128):
                distance = distance + abs((des1[i]*des1[i]) - (des2[i]*des2[i]))
            distance_arr.append((kp2_idx,distance))

        ret_values = isDistanceRatioValid(distance_arr)
        if ret_values[0]:
            matching_keypoints_1.append(keypoints_1[kp1_idx])
            matching_keypoints_2.append(keypoints_2[ret_values[1]])
            smallest_distance = ret_values[2]
            distances.append(smallest_distance)
            if smallest_distance < smallest_distance_of_all:
                smallest_distance_of_all = smallest_distance
    threshold =  8 * smallest_distance_of_all
    for idx,distance in enumerate(distances):
        if distance < threshold:
            updated_matching_keypoints_1.append(matching_keypoints_1[idx])
            updated_matching_keypoints_2.append(matching_keypoints_2[idx])
    return updated_matching_keypoints_1,updated_matching_keypoints_2


# # Step3:

# ### A

# In[44]:


def project(x1,y1,H):
    x1_y1_vector = np.array([x1,y1,1])
    projected_x1_vector = H.dot(x1_y1_vector)
    projected_x1 = projected_x1_vector[0] / projected_x1_vector[2]
    projected_y1 = projected_x1_vector[1] / projected_x1_vector[2]
    return projected_x1,projected_y1


# ### B

# In[45]:


# beware of x and y positions
# as of now considered that matches stores points in (y,x) format
# point2[1] and point2[0] would be reversed if format is (x,y)
def computeInlierCount(H, matches, inlierThreshold):
    smallest_distance = sys.maxsize
    num_of_inliers = 0
    for each_match in matches:
        y1 = each_match[1]
        x1 = each_match[0]
        y2 = each_match[3]
        x2 = each_match[2]
        projected_point_1 = project(x1,y1,H)
        x_dist = math.pow((projected_point_1[0] - x2),2)
        y_dist = math.pow((projected_point_1[1] - y2),2)
        distance = math.sqrt(y_dist+x_dist)
        
        if distance < inlierThreshold:
            num_of_inliers += 1
    
    return num_of_inliers


# In[46]:


def computeInlierMatches(H, matches, inlierThreshold):
    inlier_list = np.empty( shape=(0, 4) )
    for each_match in matches:
        y1 = each_match[1]
        x1 = each_match[0]
        y2 = each_match[3]
        x2 = each_match[2]
        projected_point_1 = project(x1,y1,H)
        x_dist = math.pow((projected_point_1[0] - x2),2)
        y_dist = math.pow((projected_point_1[1] - y2),2)
        distance = math.sqrt(y_dist+x_dist)
        
        if distance < inlierThreshold:
            inlier_list = np.append(inlier_list,each_match.reshape(-1,4),axis=0)
    return inlier_list


# ## C

# In[47]:


def generate_four_random_indexes(length):
    randomIndexes = []
    randomIndex = randint(0,length)
    while (len(randomIndexes)<4):
        randomIndexes.append(randomIndex)
        while randomIndex in randomIndexes:
            randomIndex = randint(0,length)
    return randomIndexes


# In[48]:


def RANSAC(matches, numIterations, inlierThreshold):
    # part a. of C
    best_number_of_inliers = 0
    best_hom = [[]]
    for i in range(0,numIterations):
        if len(matches) > 4:
            randomIndexes = generate_four_random_indexes(len(matches)-1)
            four_matches = matches[randomIndexes]
        elif len(matches)==4:
            four_matches = matches
            
        first_points = four_matches[:,[0,1]]
        second_points = four_matches[:,[2,3]]
        hom,_ = cv2.findHomography(first_points, second_points, 0)
        num_of_inliers = computeInlierCount(hom,matches,inlierThreshold)
        
        if num_of_inliers > best_number_of_inliers:
            best_number_of_inliers = num_of_inliers
            best_hom = hom
       
    # part b. of C
    inlier_matches = computeInlierMatches(best_hom,matches,inlierThreshold)
    
    best_hom,_ = cv2.findHomography(inlier_matches[:,[0,1]], inlier_matches[:,[2,3]], 0)
    inlier_matches_1 = inlier_matches[:,[0,1]]
    inlier_matches_2 = inlier_matches[:,[2,3]]
    homInv = np.linalg.inv(best_hom)
    return best_hom,homInv,inlier_matches_1,inlier_matches_2


# # Step4

# # a

# In[49]:


def get_projected_corners(image2,homInv):
    height = image2.shape[0]
    width = image2.shape[1]
    
    projected_point_top_left = project(0,0,homInv)
    
    projected_point_top_right = project(width-1,0,homInv)
    
    projected_point_bottom_left = project(0,height-1,homInv)
    
    projected_point_bottom_right = project(width-1,height-1,homInv)
    
    return projected_point_top_left,projected_point_top_right,projected_point_bottom_left,projected_point_bottom_right


# In[50]:


def get_height_width_of_stitched_image(image1,image2,projected_corners):
    
    height1 = image1.shape[0]
    width1 = image1.shape[1]
    
    height2 = image2.shape[0]
    width2 = image2.shape[1]
    max_height = height1 if height1 > height2 else height2
    max_width = width1 if width1 > width2 else width2
    extra_height = 0
    extra_width = 0
    extra_negative_width = 0
    extra_negative_width_final = 0
    extra_positive_width = 0
    extra_positive_width_final = 0
    extra_negative_height = 0
    extra_negative_height_final = 0
    extra_positive_height = 0
    extra_positive_height_final = 0
    for idx,each_point in enumerate(projected_corners):
        x = each_point[0]
        y = each_point[1]
        if x < 0:
            if extra_negative_width < abs(int(x)):
                extra_negative_width = abs(int(x))
                extra_negative_width_final += abs(int(x))
        elif x > max_width:
            if extra_positive_width < (int(x) - max_width):
                extra_positive_width = (int(x) - max_width)
                extra_positive_width_final += (int(x) - max_width)
        if y < 0:
            if extra_negative_height < abs(int(y)):
                extra_negative_height = abs(int(y))
                extra_negative_height_final += abs(int(y))
        elif y > max_height:
            if extra_positive_height < (int(y) - max_height):
                extra_positive_height = (int(y) - max_height)
                extra_positive_height_final += (int(y) - max_height)
    extra_height = extra_negative_height_final + extra_positive_height_final
    extra_width = extra_negative_width_final + extra_positive_width_final
    return int(max_height+extra_height),int(max_width+extra_width),int(extra_negative_height),int(extra_negative_width),int(extra_width),int(extra_height)


# In[51]:


def bilinear_interpolation(float_x,float_y,patch):
    x = int(float_x)
    y = int(float_y)
    a = float_x - x
    b = float_y - y
    
    interpolated_value = (1-a)*(1-b)*patch[0][0] + (a)*(1-b)*patch[0][1] + (1-a)*(b)*patch[1][0] + (a)*(b)*patch[1][1]
    return interpolated_value


# In[52]:


def project_image_2_on_final_image(final_image,image_2,hom,extra_height,extra_width):
    height_final_image = final_image.shape[0]
    width_final_image = final_image.shape[1]
    
    height_image_2 = image_2.shape[0]
    width_image_2 = image_2.shape[1]
    for y in range(0,height_final_image):
        for x in range(0,width_final_image):
            projected_point_x,projected_point_y = project(x-extra_width,y-extra_height,hom)
            if (projected_point_y >= 0) & (projected_point_y < height_image_2):
                if (projected_point_x >= 0) & (projected_point_x < width_image_2):
                    patch = cv2.getRectSubPix(image_2,(2,2),(projected_point_x,projected_point_y))
                    interpolated_value = bilinear_interpolation(projected_point_x,projected_point_y,patch)
                    final_image[y][x] = interpolated_value
    return final_image           


# In[53]:


def stitch(image1, image2, hom, homInv):
    # get four corners of image 2
    projected_corners = get_projected_corners(image2,homInv)
    
    height,width,extra_negative_height,extra_negative_width,extra_width,extra_height = get_height_width_of_stitched_image(image1,image2,projected_corners)
    
    # creating a blank image
    blank_image = np.zeros((int(height),int(width),3), np.uint8)
    height_1 = image1.shape[0]
    width_1 = image1.shape[1]
    
    h_min = extra_negative_height
    h_max = extra_negative_height + height_1
    w_min = extra_negative_width
    w_max = extra_negative_width+width_1
    
    blank_image[h_min:h_max,w_min:w_max] = image1
    final_image = project_image_2_on_final_image(blank_image,image2,hom,extra_negative_height,extra_negative_width)
    return final_image,extra_negative_width,extra_negative_height
   


# In[54]:


def all_images_in_panorama(image_in_panorama):
    for each_image in image_in_panorama:
        if each_image == False:
            return False
    return True


# In[55]:


def crop_image(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = gray>0
    cor_arr = np.ix_(mask.any(1),mask.any(0))
    arr_0 = cor_arr[0]
    arr_1 = cor_arr[1]

    x1 = cor_arr[0][0][0]
    y1 = cor_arr[0][-1][0]
    x2 = cor_arr[1][0][0]
    y2 = cor_arr[1][0][-1]
    img_new = img[x1:y1,x2:y2]
    return img_new


# In[75]:


def make_panorama_of_two_images(img_1,img_2,i):
    if i==1:
        img_1_copy = img_1.copy()
        img_2_copy = img_2.copy()
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptor1 = sift.detectAndCompute(img_1,None)
    keypoints2, descriptor2 = sift.detectAndCompute(img_2,None)

    bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=False)
    matches = bf.match(descriptor1, descriptor2)
    matches = sorted(matches, key = lambda x:x.distance)

    matching_kps_1,matching_kps_2 = matchFeatures(descriptor1,descriptor2,keypoints1,keypoints2,matches[:50])
    matching_kp_1 = np.float32([m.pt for m in matching_kps_1])
    matching_kp_2 = np.float32([m.pt for m in matching_kps_2])
    
    matches = np.concatenate((matching_kp_1, matching_kp_2), axis=1)
    numIterations = 500
    inlierThreshold = 0.5
    hom,homInv,inlier_matches_1,inlier_matches_2 = RANSAC(matches, numIterations, inlierThreshold)
    if i==1:
        create_keypoints_matching_image(img_1_copy, img_2_copy,inlier_matches_1,inlier_matches_2,"3.png",1)
        print('3.png has been stored in current directory depicting inlier matches between Rainier1.png and Rainier2.png')
    no_of_inliers = len(inlier_matches_1) 
    return hom,homInv,no_of_inliers


# In[66]:


# stitching Rainier1.png to Rainier6.png
def make_panorama(directory):
    no_of_images = len(os.listdir(directory))
    image_in_panorama = [False] * no_of_images
    image_paths = []
    match_image_store = -1
    for each_image in os.listdir(directory):
        image_paths.append(os.path.join(directory,each_image))
    if no_of_images == 2:
        if ('Rainier1.png' in image_paths[0]) | ('Rainier2.png' in image_paths[1]):
            match_image_store = 1
        elif ('Rainier2.png' in image_paths[0]) | ('Rainier1.png' in image_paths[1]):
            match_image_store = 1
    image_in_panorama[0] = True
    print(image_paths[0],'has been added to the final image')
    img_1 = cv2.imread(image_paths[0])
    final_image = img_1
    count = 1
    while not all_images_in_panorama(image_in_panorama):
        count += 1
        no_of_inliers = 0
        candidate_image_index = 0
        best_hom = []
        best_homInv = []
        for j in range(1,no_of_images):
            if not image_in_panorama[j]:
                img_2 = cv2.imread(image_paths[j])
                hom,homInv,local_no_of_inliers = make_panorama_of_two_images(final_image,img_2,match_image_store)
                if local_no_of_inliers > no_of_inliers:
                    no_of_inliers = local_no_of_inliers
                    candidate_image_index = j
                    best_hom = hom
                    best_homInv = homInv
        print(image_paths[candidate_image_index],'will next be stitched to the final image')
        image_in_panorama[candidate_image_index] = True
        img_2 = cv2.imread(image_paths[candidate_image_index])
        
        final_image,extra_negative_width,extra_negative_height = stitch(final_image, img_2, best_hom, best_homInv)
        
    return final_image


# In[67]:


print('\n')
print('Making panorama of Rainier1.png and Rainier2.png')
stitched_image_2 = make_panorama("project_images\\Rainier_2_images\\")
stitched_image_2 = crop_image(stitched_image_2)
cv2.imwrite('4.png',stitched_image_2)
print('4.png has been stored in current directory depicting stitched image (panorama) of images in project_images\Rainier_2_images folder')


# In[68]:


print('\n')
print('Making panorama of all images in project_images\Rainier_6_images folder')
stitched_image_6 = make_panorama('project_images\\Rainier_6_images\\')
stitched_image_6 = crop_image(stitched_image_6)
cv2.imwrite('my_AllStitched.png',stitched_image_6)
print('my_AllStitched.png has been stored in current directory depicting stitched image (panorama) of all images in project_images\Rainier_6_images folder')


# In[69]:


# stitching my clicked images
print('\n')
print('Making panorama of all images in project_images\My_clicked_images folder')
my_stitched_image = make_panorama('project_images\\My_clicked_images\\')
my_stitched_image = crop_image(my_stitched_image)
cv2.imwrite('my_clicked_images_stitched.png',my_stitched_image)
print('my_clicked_images_stitched.png has been stored in current directory depicting stitched image (panorama) of all images in project_images\My_clicked_images folder')


# In[70]:


print('\n')
x = input('Do you want to make more panoramas? Enter y if yes.')
while x.lower() == 'y':
    input_dir = input('Put the images in a folder in current directory and enter the name of the folder you created.')
    my_stitched_image = make_panorama(input_dir)
    my_stitched_image = crop_image(my_stitched_image)
    cv2.imwrite('my_custom_images_stitched.png',my_stitched_image)
    print('my_custom_images_stitched.png has been stored in current directory depicting stitched image (panorama) of all the images in',input_dir,'folder')
    x = input('Do you want to make more panoramas? Enter y if yes else press any other key')


# In[ ]:





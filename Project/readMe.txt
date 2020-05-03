To run the code :
-> Run 'code.py' as a normal python file.
-> All required images have been added to the project_images folder.
-> All result images have been stored in 'Result_images' folder.
-> All the required output images would be stored in the current directory.
-> At the end, the execution will prompt an input box. Answer 'y', if you want to create another panorama.
-> If you entered 'y', you will be prompted for a folder name.
    -> In case, you want to create panorma for the Melakwala_lake images, enter 'Melakwala_images' in the input box. (This folder with the 
        relevant images is already present in the current directory)
    -> Skip the next line, if you went by the option of Melakwala lake images.
        -> Before entering the folder name, create that folder in the current directory and add the images whose panorama you want to create to it.
-> The stitched image or panorama would be stored in the current directory under the name 'my_custom_images_stitched.png'.

Features Implemented:
1. Feature Detection
• Here, I applied the Harris corner detection algorithm with a threshold of 0.4 times the maximum corner response for any potential key-point in the image.
• Then, I selected interest points which were local maximum in a 3x3 neighborhood
• After that, I selected interest point using adaptive non maximum suppression to ensure uniform spatial spread of the key-points.
• At this step, dominant orientation for each key-point was calculated and also, those key-points with 80% of the maximum magnitude were considered and added to the list of key-points.
• This feature has been applied on Boxes.png, Rainier1.png and Rainier2.png and corresponding images would be stored in the current directory as 1a.png, 1b.png and 1c.png.

2. Feature Description and Matching
• 128 bit SIFT descriptor was calculated here and the descriptor was normalized to ensure contrast invariance.
• A threshold of 1.8 times the smallest distance between two features has been considered.
• Ratio test was used to further filter better matches.
• An image '2.png' is stored in the current directory showing matches between Rainier1.png and Rainier2.png


Before going through the functions of 3rd and 4th steps, I would explain the flow of the rest of the project which is as follows:

• The image match pair which is to be included first in the panorama is found out, depending on the number of inliers using RANSAC algorithm and 
    likewise, is done for the remaining images.
• The keypoint matches used as input for RANSAC algorithm are found in the following way:
    • Inbuilt SIFT detector has been used to get key-points and descriptors of those key-points.
    • An inbuilt Brute Force Matcher (BFMatcher) is used to match the descriptors depending on SSD.
    • Since, matches obtained this way are large in number, best matches are selected from the top 50 matches using ratio test where a 
        threshold of 8 times the smallest distnace between any match, is used.
• After the image_2 to be added to panorama has been identified, it is stitched with image_1 using the following steps:
    • Four projected corners of the image_2 are found out using the given homography.
    • Then, the height and width of the panorama image is calculated.
    • Image_1 is copied at a proper location on the final panorama image.
    • Image_2 is projected on the final panorama image using bilinear_interpolation.
• In case of more than two images, the first three steps are performed until all the images have been added to the final panorama image.

3. Compute Homography using RANSAC
    Mandatory Functions:
    1. Project(...) : This helps to find projection of a given point using a given homography.
    2. computeInlierCount(...) : This helps to count the number of inliers given an inlier threshold, matches and homography.
    3. RANSAC(...) : This helps to find the best homography, given matches between two images, number of iterations indicating the number 
        of times to run the RANSAC algorithm and inlier threshold (points having inlier distance less than this threshold would be 
        considered as inliers). In this project, I used number of iterations = 500 and inlier threshold = 0.5.
    Other Helper Functions:
    1. computeInlierMatches(...) : This helps to compute the final set of inliers after the best homography is found by RANSAC.
    2. generate_four_random_indexes(...) : This helps to generate random 4 indexes covered by a given number.
    Other Comments:
    • An image '3.png' is stored in the current directory showing inlier matches between Rainier1.png and Rainier2.png

4. Image Stitching
    Mandatory Functions:
    1. stitch(...) : This function is responsible for stitching two images and form a panorama.
    Other Helper Functions:
    1. get_projected_corners(...) : to get projected corners of the second image using inverse homography
    2. bilinear_interpolation(...) : to calculate the interpolated RGB values for pixels given the floating point projected points and 
        RGB values of pixels of a 2 x 2 patch around the floating point
    3. project_image_2_on_final_image(...) : to project image 2 on the stitched image where image 1 had already been copied before starting 
        to project image 2.
    4. get_height_width_of_stitched_image(...) : to calculate the height and width of the stitched image.
    Other Comments:
    • An image '4.png' is stored in the current directory showing panorama including images Rainier1.png and Rainier2.png

Mandatory Extra Credits:
1. Panorama of 6 images:
    Functions:
    1. make_panorama_of_two_images(...) : given two images, it will run RANSAC and find the number of inliers between those images.
    2. make_panorama(...) : given the name of the directory containing valid images for a single panorama, this function would find the
        order in which these images should be added to the final image by first adding the images with highest number inliers and likewise
        for the remaining images.
    3. all_images_in_panorama(...) : checks if all images have been added to the panorama.
    4. crop_image(...) : to remove the extra height and width of the image
    Other Comments:
    • An image 'my_AllStitched.png' is stored in the current directory showing panorama including Rainier1.png, Rainier2.png, Rainier3.png, 
        Rainier4.png, Rainier5.png and Rainier6.png.

2. Panorama of my clicked images:
    • Under this, I have prepared a panorama of 3 images clicked by me, which have been stored in 'my_clicked_images' folder.
    • An image 'my_clicked_images_stitched.png' is stored in the current directory showing panorama of images in 'project_images\My_clicked_images' folder.

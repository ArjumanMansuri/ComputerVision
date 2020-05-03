To run the code :
-> Please put the 'image_sets' folder provided in the assignment as a zip file, at the same level as the python file.
-> Please enter image path as 'yosemite\Yosemite1.jpg' and 'yosemite\Yosemite2.jpg' when prompted.

Features Implemented:
1. Feature Detection
• Here, I applied the Harris corner detection algorithm with a threshold of 0.4 times the maximum corner response for any potential key-point in the image.
• Then, I selected interest points which were local maximum in a 3x3 neighborhood
• After that, I selected interest point using adaptive non maximum suppression to ensure uniform spatial spread of the key-points.
• At this step, dominant orientation for each key-point was calculated and also, those key-points with 80% of the maximum magnitude were considered and added to the list of key-points.
2. Feature Description
• 128 bit SIFT descriptor was calculated here and the descriptor was normalized to ensure contrast invariance.
3. Feature Matching
• A threshold of 1.8 times the smallest distance between two features has been considered.
• Ratio test was used to further filter better matches.
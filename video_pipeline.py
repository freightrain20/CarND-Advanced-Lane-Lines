import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# Read in the saved camera matrix and distortion coefficients
dist_pickle = pickle.load( open( "camera_cal_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

global left_curverad_prev
global right_curverad_prev

left_curverad_prev = 0
right_curverad_prev = 0
# Unwarp the image
def unwarp(img, mtx, dist):
    
    img_u = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(img_u, cv2.COLOR_BGR2GRAY)
    
    # Example Images source and destination coordinates for image warp
    #src = np.float32([[397, 365], [573, 365], [160, 530], [840, 530]])
    #dst = np.float32([[100, 100], [700, 100], [100, 500], [700, 500]])
    
    # Test Images and Project Video source and destination coordinates for image warp
    src = np.float32([[558, 475], [757, 475], [225, 700], [1200, 700]])
    dst = np.float32([[100, 100], [1100, 100], [100, 700], [1100, 700]])
    
    # Challenge Video source and destination coordinates for image warp
    #src = np.float32([[570, 490], [758, 490], [225, 700], [1200, 700]])
    #dst = np.float32([[100, 100], [1100, 100], [100, 700], [1100, 700]])
    
    # Perspective transformation
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_u, M, gray.shape[::-1], flags = cv2.INTER_LINEAR)
    
    return warped, M

# Apply sobel gradient in x or y direction
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output
    
# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
# Not used in the final pipeline
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output
    
# Define a function to threshold an image for a given range and Sobel kernel
# Not used in the final pipeline
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image based on color hue, saturation, and/or lightness (HLS space)    
def hls_select(img, min_thresh=(0, 0, 0), max_thresh=(255, 255, 255)):
    
    # Convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    
    # Create a binary image where HLS values are within predefined thresholds
    binary_output = np.zeros_like(H)
    binary_output[(H > min_thresh[0]) & (H <= max_thresh[0]) & (L > min_thresh[1]) & (L <= max_thresh[1]) & (S > min_thresh[2]) & (S <= max_thresh[2])] = 1
    
    # Return the binary image
    return binary_output

# Image processing pipeline
def process_image(img):
    # Global variables to store (x,y) location of lane lines
    global y_val_L_prev
    global x_val_L_prev
    global y_val_R_prev
    global x_val_R_prev
    global left_curverad_prev
    global right_curverad_prev
    
    # Yellow and white HLS thresholds
    max_thresh_y = (50, 190, 255)
    min_thresh_y = (15, 140, 100)
    max_thresh_w = (255, 255, 255)
    min_thresh_w = (0, 200, 0)
    
    # Unit conversion from pixels to meters
    #ym_per_pix = 3/150 # meters per pixel in y dimension
    #xm_per_pix = 3.7/850 # meteres per pixel in x dimension
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Main pipeline process
    # 1. Change image to overhead perspective
    # 2. X direction sobel gradient
    # 3. Y direction sobel gradient
    # 4. HLS filter for yellow lines
    # 5. HLS filter for white lines
    # 6. Combine various filters into one binary image
    # 7. Moving histogram window to identify left and right lane lines
    # 8. Fit a polynomial to estimated left and right lane markings
    # 9. Calculate radius of curvature
    # 10. Calculate the vehicle position in the lane
    
    # 1. Change image to overhead perspective
    top_down, perspective_M = unwarp(img, mtx, dist)
    
    # Calculate inverse to warp image back to original perspective
    Minv = np.linalg.inv(perspective_M)
    
    # 2. X direction sobel gradient
    grad_x_binary = abs_sobel_thresh(top_down, orient='x', thresh_min=20, thresh_max=170)
    
    # 3. Y direction sobel gradient
    grad_y_binary = abs_sobel_thresh(top_down, orient='y', thresh_min=20, thresh_max=100)
    
    # Magnitude and direction gradients not used in final pipeline
    #mag_binary = mag_thresh(top_down, sobel_kernel=9, mag_thresh=(30, 190))
    #dir_binary = dir_threshold(top_down, sobel_kernel=15, thresh=(0.7, 1.2))
    
    # 4. HLS filter for yellow lines
    hls_binary_yellow = hls_select(top_down, min_thresh_y, max_thresh_y)
    
    # 5. HLS filter for white lines
    hls_binary_white = hls_select(top_down, min_thresh_w, max_thresh_w)
    
    # 6. Combine various filters into one binary image
    combined = np.zeros_like(grad_x_binary)
    combined[((grad_x_binary == 1) & (grad_y_binary == 1)) | (hls_binary_yellow == 1) | (hls_binary_white == 1)] = 1
      
    # 7. Moving histogram window to identify left and right lane lines
    # Each histogram consists of half of the image space, eight histograms in total
    # The max value in the histogram on the left half and right half of the image are assumed to be lane lines
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(combined[combined.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((combined, combined, combined))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(combined.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = combined.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = combined.shape[0] - (window+1)*window_height
        win_y_high = combined.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, combined.shape[0]-1, combined.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # 9. Calculate a radius of curvature by fitting a 2nd order polynomial half way between the left and right lane polynomials
    # Fit new polynomials to x,y in world space
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    if left_curverad_prev:
        left_curverad = round(0.1 * left_curverad + 0.9 * left_curverad_prev)
        right_curverad = round(0.1 * right_curverad + 0.9 * right_curverad_prev)
    
    left_curverad_prev = left_curverad
    right_curverad_prev = right_curverad
    
    # 10. Calculate the vehicle position within the lane
    # 530 pixels is the location of the center of the image, warped
    center_of_lane = 530 * xm_per_pix
    # Add half of the lane width to the left lane polynomial to estimate the vehicle location
    center_offset = round((center_of_lane - (left_fitx[-1] * xm_per_pix + 3.7 / 2)), 1)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(combined).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    # Write radius of curvature and lane location to image
    cv2.putText(result, 'Radius of Curvature: ' + str(left_curverad) + ' m (left) ' + str(right_curverad) + ' m (right)', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    if center_offset >= 0:
        cv2.putText(result, 'Distance from Center of Lane: ' + str(center_offset) + ' m to the right', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    else:
        cv2.putText(result, 'Distance from Center of Lane: ' + str(-center_offset) + ' m to the left', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    
    # Return the final image
    return result

# Video processing
white_output = 'project_video_labeled_v3.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
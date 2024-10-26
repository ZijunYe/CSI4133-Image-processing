# Lab4
# Name: Zijun Ye 
# Student Number: 300168065
import cv2 
import numpy as np

def nothing(x):
    pass

#1 load image 
pic1 = cv2.imread('images/Picture2.png')
pic2 = cv2.imread('images/Picture3.png')

pic1_hsv = cv2.cvtColor(pic1, cv2.COLOR_BGR2HSV)
pic2_hsv = cv2.cvtColor(pic2, cv2.COLOR_RGB2HSV)

cv2.imshow("hsv Image",pic2_hsv)
# Part A 
# create track bar 
cv2.namedWindow('Processed Hue')

# quit the loop if user pressed key 
def update_hue(val): 
    # get current hue value that show in the trackbar
    hue_value = cv2.getTrackbarPos('Hue','Processed Hue')

    daylight = pic2_hsv.copy()
    # loodp through each pixel to get their hue value that is matches for current hue_value in the bar 
    for x in range(daylight.shape[0]): 
        for y in range(daylight.shape[1]): 
            h,s,v = daylight[x,y]
            # if current current pixel's hue value =  hue value, set to HSV value all to 0 
            if h != hue_value: 
                daylight[x,y]=[0,0,0]
    # convert the hue to bgr image 
    result = cv2.cvtColor(daylight,cv2.COLOR_HSV2RGB)
    
    cv2.imshow("Processed Hue",result)


cv2.createTrackbar('Hue', 'Processed Hue', 0, 179, update_hue)



# Part B 


cv2.namedWindow('Color Detection')

daylight2 = pic2_hsv.copy()

# Define a proper color range - Min and max 

# Yellow_Green --> Hue value 30 - 80 
yellow_green_lower = np.array([80, 100, 100])  # Yellow-green lower bound
yellow_green_upper = np.array([86, 255, 255])  # Yellow-green upper bound
# Violet --> 140 - 160 
violet_lower = np.array([150, 0, 0])  # Violet lower bound
violet_upper = np.array([154, 255, 255])  # Violet upper bound

# Red   --> part1 0 - 10, part2 165 - 180
red_lower1 = np.array([118, 100, 50])  # Red lower bound (low range)
red_upper1 = np.array([120, 255, 255])  # Red upper bound (low range)
red_lower2 = np.array([170, 100, 100])  # Red lower bound (high range)
red_upper2 = np.array([179, 255, 255])  # Red upper bound (high range)


def refine_mask(lower_bound, upper_bound, hsv_img):
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

    kernel = np.ones((3, 3), np.uint8)
    
    # # Erosion (remove noise)
    # eroded_mask = cv2.erode(mask, kernel, iterations=1)
    
    # # # Dilation (fill in gaps)
    # dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    
    return mask

def update_mask(val): 
    selected_color = cv2.getTrackbarPos('selector', 'Color Detection')

    if selected_color == 0:  # Yellow-Green selected
        mask = refine_mask(yellow_green_lower, yellow_green_upper, daylight2)
    elif selected_color == 1:  # Violet selected
        mask = refine_mask(violet_lower, violet_upper, daylight2)
    elif selected_color == 2:  # Red selected
        mask1 = refine_mask(red_lower1, red_upper1, daylight2)
        mask2 = refine_mask(red_lower2, red_upper2, daylight2)
        mask = cv2.bitwise_or(mask1, mask2)


    result = cv2.bitwise_and(pic2, pic2, mask=mask)

    cv2.imshow('Color Detection', result)


# Create the trackbar (0 for Yellow-Green, 1 for Violet, 2 for Red)
cv2.createTrackbar('selector', 'Color Detection', 0, 2, update_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()

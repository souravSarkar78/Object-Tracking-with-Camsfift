# import necessary packages
import cv2
import numpy as np



# Naming the Output window
windowname = 'Result'
cv2.namedWindow(windowname)

cap = cv2.VideoCapture(0)

output = None

x, y, w, h = 0, 0, 0, 0

point_saved = False
counter = 0
track_window = (x, y, w, h)
can_track = False

def click_event(event, px, py, flags, param):
    global x, y, w, h, point_saved, track_window, counter, can_track, output

    if event == cv2.EVENT_LBUTTONUP:
        if point_saved:
            w = px-x
            h = py-y
            track_window = (x, y, w, h)
            print(x, y, w, h)
            point_saved = False
            counter = 2
        else:
            x = px
            y = py
            point_saved = True
            can_track = False
            counter = 1
            

        
    if event == cv2.EVENT_RBUTTONDOWN:
        can_track = False
cv2.setMouseCallback(windowname, click_event)  # Start the mouse event



# initialize tracker frame

def initialize(frame, track_window):
    x, y, w, h = track_window
    # set up the ROI for tracking
    roi = frame[y:y+h, x:x+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    return roi_hist, roi

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Check id 2nd point is also selected then reinitialize everything
    if counter == 2:
        roi_hist, roi = initialize(frame, track_window)
        counter = 0
        can_track = True
    
    # Start tracking
    if can_track == True:
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply camshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        print(track_window)
        cv2.imshow('roi', roi)
        output = cv2.polylines(frame,[pts],True, 255,2)
        
    else:
        output = frame
        if counter == 1:
            cv2.circle(output, (x, y), 3, (0, 255, 0), -1)
        cv2.destroyWindow('roi')
        

    cv2.imshow(windowname,output)
    if cv2.waitKey(1) == ord('q'):
        break
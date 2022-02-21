import pyrealsense2 as rs
import numpy as np
import cv2
import imutils
import math
f=open("coordinate.txt","w")
# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)
# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
#clipping_distance_in_meters = 100 #1 meter
#clipping_distance = clipping_distance_in_meters / depth_scale
# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue
        coverage = [0]*64
        for y in range(480):
            for x in range(640):
                dist = depth.get_distance(x, y)
                if x==320 and y==240:
                    distS=dist
                    dist=round(dist,2)
                    f.write("distanza"+str(dist)+" ")
                if 0 < dist and dist < 1:
                    coverage[x//10] += 1

        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())  
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV )
        lower=np.array([36,0,0]) # Il minimo del colore
        upper=np.array([85,255,255]) # Il massimo del colore
        k=cv2.inRange(hsv,lower,upper) # Tramite il massimo ed il minimo del colore la trova nell'immagine 
        cv2.imshow('lll',k)
        k=cv2.erode(k,None,iterations=1) # Migliora il controno del colore selezionato
        cvst=cv2.findContours(k.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # trova il contorno 
        cvst=imutils.grab_contours(cvst) 
        clone=color_image.copy()
        cv2.drawContours(clone,cvst,-1,(0,255,0),3) # Disegna i contorni   
        max=0
        (cxMax,cyMax)=(0,0)
        for c in cvst:
            x,y,w,h= cv2.boundingRect(c)
            (j,g)=(x+w,y+h)
            (centroX,centroY)=((j+x) // 2,(g+y) // 2)
            if(max<(g-y)):
                max=(g-y)
                (cxMax,cyMax)=(centroX,centroY)
        if (cxMax,cyMax)!=(0,0):          
            cv2.circle(clone,(cxMax,cyMax),10,(0,0,0),-1)
        else:
            print("non trovo l'oggetto")    
        cv2.imshow('frame',clone)
        cxS = (cxMax)
        cyS = (cyMax)
        if(cxS<320):
            dpx=320-cxS
            x=-dpx*(620/int(distS))
        else:
            dpx=cxS-320
            x=dpx*(620/int(distS))
        if(cyS<240):
            dpy=240-cyS
            x=-dpy*(480/int(distS))
        else:
            dpy=cyS-240
            x=dpy*(480/int(distS))
        f.write("x:"+str(x)+" y:"+str(cyS)+"\n")
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()




f.close()
import os
import subprocess
import numpy as np
import cv2

dirs = set([])
video_ids = ["01"]
root_dir = "chollec80_raw_data"
destination_dir = "chollec80_processed_data"

#loop through all  videos

for video_id in video_ids:

    subprocess.call(["rm", "-rf", "proc"])
    subprocess.call(["mkdir", "proc"])

    #video preprocess (should be higher dimension, but cut like this for now)
    subprocess.call(["ffmpeg", "-i", root_dir + "/video" + video_id + ".mp4", "-r", "25", "-s", "224x224", "-aspect", "4:3", "proc/video_process.mp4"])

    for min in range(0, 25, 1):
        for sec in range(0, 60, 10):

            subprocess.call(["rm", "-rf", "tmp"])
            subprocess.call(["mkdir", "tmp"])

            if ( (sec+10) == 60 ):
                next_min = min+1
                next_sec = 0
            else:
                next_min = min
                next_sec = sec+10

            # cut videos in 10s samples
            subprocess.call(["ffmpeg", "-i", "proc/video_process.mp4", "-ss", "00:"+('%02d' % min)+":"+('%02d' % sec), "-to", "00:"+('%02d' % next_min)+":"+('%02d' % next_sec), "tmp/video_process_cropped.mp4"])


            #sample the vids into 25 fps
            subprocess.call(["ffmpeg", "-i", "tmp/video_process_cropped.mp4", "tmp/thumb%04d.jpg"])

            #loop through all images, 25 fps for 10s
            for i in range(1, 251):

                image_str = "tmp/thumb" + ('%04d' % i) + ".jpg"
                image = cv2.imread(image_str, cv2.IMREAD_COLOR)

                #handle RGB
                norm_image_rgb = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                #Crop 224x224 samples

                #reshape
                norm_image_rgb_r = norm_image_rgb.reshape(1, 224, 224, 3)

                #Add to RGB numpy
                if ( i == 1 ):
                    rgb_video = norm_image_rgb_r
                else:
                    rgb_video = np.concatenate((rgb_video, norm_image_rgb_r), axis=0)

                #Flow

                #Convert to Greyscale
                if ( i == 1 ):
                    prev_image = image

                gray_image_prev = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                #Run Optical Flow
                dtvl1 = cv2.createOptFlow_DualTVL1()
                image_flowDTVL1 = dtvl1.calc(gray_image_prev, gray_image, None)

                #Truncate Pixels between [-20, 20]
                #image_flow1, imageflow2 = cv2.split(image_flowDTVL1)
                image_flow_truncated = np.clip(image_flowDTVL1, -20, 20)

                #Normalize image into [-1, 1]
                norm_image_flow = cv2.normalize(image_flow_truncated, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                #Crop 224x224 samples

                #reshape
                norm_image_flow_r = norm_image_flow.reshape(1, 224, 224, 2)

                #Add to Flow numpy
                if ( i == 1 ):
                    flow_video = norm_image_flow_r
                else:
                    flow_video = np.concatenate((flow_video, norm_image_flow_r), axis=0)

                prev_image = image

                print ("Processed image " + str(i))


            rgb_video_r = rgb_video.reshape(1, 250, 224, 224, 3)
            flow_video_r = flow_video.reshape(1, 250, 224, 224, 2)

            #when done, copy .npy files for one 10s video clip, RGB and Flow
            destination_rgb = destination_dir + "/video_rgb" + video_id + "_" + str(min) + str(sec) + ".npy"
            destination_flow = destination_dir + "/video_flow" + video_id + "_" + str(min) + str(sec) + ".npy"
            np.save(destination_rgb, rgb_video_r)
            np.save(destination_flow, flow_video_r)

	    print("Processed video " + str(min) + " " + str(sec))

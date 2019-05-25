import os
import subprocess
import numpy as np
import cv2

def run_processing():

    dirs = set([])
    video_ids = ["61", "62", "63", "64", "65", "66", "67", "68", "69", "70",
                 "71", "72", "73", "74", "75", "76", "77", "78", "79", "80"]
    root_dir = "chollec80_raw_data"
    destination_dir = "chollec80_processed_data_full"

    list_file_txt_rgb = "chollec80_processed_list_test_rgb_full.txt"
    list_file_txt_flow = "chollec80_processed_list_test_flow_full.txt"
    subprocess.call(["rm", "-rf", "chollec80_processed_list_test_rgb_full.txt"])
    subprocess.call(["rm", "-rf", "chollec80_processed_list_test_flow_full.txt"])
    list_file_rgb = open(list_file_txt_rgb,"w+")
    list_file_flow = open(list_file_txt_flow,"w+")

    #loop through all  videos

    for video_id in video_ids:

        subprocess.call(["rm", "-rf", "proc"])
        subprocess.call(["mkdir", "proc"])

        label_txt = root_dir + "/video" + video_id + "-timestamp.txt"
        label_f = open(label_txt, "r")
        label_f.readline()

        #video preprocess (should be higher dimension, but cut like this for now)
        subprocess.call(["ffmpeg", "-i", root_dir + "/video" + video_id + ".mp4", "-r", "25", "-s", "224x224", "-aspect", "4:3", "proc/video_process.mp4"])

        #get hour and minute bounds
        time_limit_list = list(open(label_txt, 'r'))
        time_limit_len = len(time_limit_list)

        time_limit_f = open(label_txt, "r")

        for l in range(1, time_limit_len)
            time_limit_f.readline()

        limit_line = time_limit_f.readline()
        last_time = limit_line.split('    ')[0][:-1]

        print (last_time)

        for hour in range(0, hour_max, 1):
            for minute in range(0, min_max, 2):

                subprocess.call(["rm", "-rf", "tmp"])
                subprocess.call(["mkdir", "tmp"])

                if ( (min+2) == 60 ):
                    next_hour = 1
                    next_min = 0
                else:
                    next_hour = hour
                    next_min = min+2

                # cut videos in 10s samples
                subprocess.call(["ffmpeg", "-i", "proc/video_process.mp4", "-ss", ('%02d' % hour)+":"+('%02d' % minute)+":"+"00", "-to", ('%02d' % next_hour)+":"+('%02d' % next_min)+":"+"10", "tmp/video_process_cropped.mp4"])


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
                   
        		    if ( i == 1 ):    
        		    	image_flowDTVL1 = dtvl1.calc(gray_image_prev, gray_image, None)
        		    else:
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
                destination_rgb = destination_dir + "/video_rgb" + video_id + "_" + str(hour) + str(minute) + ".npy"
                destination_flow = destination_dir + "/video_flow" + video_id + "_" + str(hour) + str(minute) + ".npy"
                np.save(destination_rgb, rgb_video_r)
                np.save(destination_flow, flow_video_r)

                print("Processed video " + str(hour) + " " + str(minute))

                line = label_f.readline()

                if line != '\n':
                    raw_label = line.split('	')[1][:-1]
                else:
                    raw_label = 6

                label = label_decoder(raw_label)
                label_title_rgb = "video_rgb" + video_id + "_" + str(hour) + str(minute)
                label_title_flow = "video_flow" + video_id + "_" + str(hour) + str(minute)
		        list_file_rgb.write(label_title_rgb)
                list_file_rgb.write('	')
                list_file_rgb.write(str(label))
                list_file_rgb.write("\n")
        		list_file_flow.write(label_title_flow)
        		list_file_flow.write('	')
        		list_file_flow.write(str(label))
        		list_file_flow.write("\n")

                #skip until the next label
                #for this, two minutes, ideally make this variable length
                for t in range(1, 3000):
                    label_f.readline()


    list_file_rgb.close()
    list_file_flow.close()


def label_decoder(raw_label):

    if(raw_label == "Preparation"):
        return 0
    elif(raw_label == "CalotTriangleDissection"):
        return 1
    elif(raw_label == "ClippingCutting"):
        return 2
    elif(raw_label == "GallbladderDissection"):
        return 3
    elif(raw_label == "GallbladderPackaging"):
        return 4
    elif(raw_label == "CleaningCoagulation"):
        return 5
    elif(raw_label == "GallbladderRetraction"):
        return 6


def main(_):
    run_processing()

if __name__ == '__main__':
    run_processing()

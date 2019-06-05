import os
import subprocess
import numpy as np
import cv2

def run_processing():

    dirs = set([])
    video_ids = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", 
                 "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                 "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
                 "31", "32", "33", "34", "35", "36", "37", "38", "39", "40"
                 "41", "42", "43", "44", "45", "46", "47", "48", "49", "50"
                 "51", "52", "53", "54", "55", "56", "57", "58", "59", "60"]
    root_dir = "chollec80_raw_data"
    destination_dir = "chollec80_processed_data_full_batch2"

    #loop through all  videos

    for video_id in video_ids:

        subprocess.call(["rm", "-rf", "proc"])
        subprocess.call(["mkdir", "proc"])

        #video preprocess (should be higher dimension, but cut like this for now)
        subprocess.call(["ffmpeg", "-i", root_dir + "/video" + video_id + ".mp4", "-r", "5", "-s", "224x224", "-aspect", "4:3", "proc/video_process.mp4"])

        label_txt = root_dir + "/video" + video_id + "-timestamp.txt"

        #get hour and minute bounds
        time_limit_list = list(open(label_txt, 'r'))
        time_limit_len = len(time_limit_list)

        time_limit_f = open(label_txt, "r")

        for l in range(1, time_limit_len):
            time_limit_f.readline()

        limit_line = time_limit_f.readline()
        info = limit_line.split()
        last_time = info[0]

        last_hour = int(last_time[0] + last_time[1])
        last_min = int(last_time[3] + last_time[4])

        for hour in range(0, (last_hour+1), 1):
            for minute in range(0, (60), 1):
                for second in range(0, (60), 30):

                    subprocess.call(["rm", "-rf", "tmp"])
                    subprocess.call(["mkdir", "tmp"])

                    next_hour = hour
                    next_min = minute
                    next_second = second+10

                    # cut videos in 10s samples
                    subprocess.call(["ffmpeg", "-i", "proc/video_process.mp4", "-ss", ('%02d' % hour)+":"+('%02d' % minute)+":"+('%02d' % second), "-to", ('%02d' % next_hour)+":"+('%02d' % next_min)+":"+('%02d' % next_second), "tmp/video_process_cropped.mp4"])


                    #sample the vids into 5 fps
                    subprocess.call(["ffmpeg", "-i", "tmp/video_process_cropped.mp4", "tmp/thumb%04d.jpg"])

                    #loop through all images, 5 fps for 10s
                    for i in range(1, 51):

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

                        print ("Processed image " + str(i))


                    rgb_video_r = rgb_video.reshape(1, 50, 224, 224, 3)

                    #when done, copy .npy files for one 10s video clip, RGB and Flow
                    destination_rgb = destination_dir + "/video_rgb" + video_id + "_" + str(hour) + str(minute) + str(second) + ".npy"
                    np.save(destination_rgb, rgb_video_r)

                    print("Processed video " + str(video_id) + " " + str(hour) + " " + str(minute) + " " + str(second))

                    if (((minute+1) >= last_min) and (hour == last_hour)):
                        break


def main(_):
    run_processing()

if __name__ == '__main__':
    run_processing()

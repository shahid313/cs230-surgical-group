import os
import subprocess
import numpy as np
import cv2

def run_processing():

    dirs = set([])
    video_ids = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", 
                 "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                 "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
                 "31", "32", "33", "34", "35", "36", "37", "38", "39", "40",
                 "41", "42", "43", "44", "45", "46", "47", "48", "49", "50",
                 "51", "52", "53", "54", "55", "56", "57", "58", "59", "60"]
    root_dir = "chollec80_raw_data"

    list_file_txt_rgb = "chollec80_processed_list_rgb_full_batch2.txt"
    list_file_txt_flow = "chollec80_processed_list_flow_full_batch2.txt"
    subprocess.call(["rm", "-rf", list_file_txt_rgb])
    subprocess.call(["rm", "-rf", list_file_txt_flow])
    list_file_rgb = open(list_file_txt_rgb,"w+")
    list_file_flow = open(list_file_txt_flow,"w+")

    #loop through all  videos

    for video_id in video_ids:

        label_txt = root_dir + "/video" + video_id + "-timestamp.txt"
        label_f = open(label_txt, "r")
        label_f.readline()

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

                    line = label_f.readline()

                    if line != '\n':
                        raw_label = line.split('	')[1][:-1]
                    else:
                        raw_label = 6

                    label = label_decoder(raw_label)
                    label_title_rgb = "video_rgb" + video_id + "_" + str(hour) + "_" + str(minute) + "_" + str(second)
                    label_title_flow = "video_flow" + video_id + "_" + str(hour) + "_" + str(minute) + "_" + str(second)
                    list_file_rgb.write(label_title_rgb)
                    list_file_rgb.write('	')
                    list_file_rgb.write(str(label))
                    list_file_rgb.write("\n")
                    list_file_flow.write(label_title_flow)
                    list_file_flow.write('	')
                    list_file_flow.write(str(label))
                    list_file_flow.write("\n")

                    #skip until the next label
                    #for this, 30 sec, ideally make this variable length
                    for t in range(1, 750):
                        label_f.readline()

                    #if true, exit the loops

                    if (((minute+1) >= last_min) and (hour == last_hour) and (second == 30)):
                        video_done = True
                        break
                if (video_done == True):
                    break
            if (video_done == True):
                break


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

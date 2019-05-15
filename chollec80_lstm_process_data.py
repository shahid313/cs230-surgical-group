import subprocess 

dirs = set([])
video_ids = ["14"] #TODO add all video ids that you want to process
root_dir = "chollec80_raw_data"
destination_dir "chollec80_processed_data"


for video_id in video_ids:
	subprocess.call(["rm", "-rf", "tmp"])
	subprocess.call(["mkdir", "tmp"])
	subprocess.call(["ffmpeg", "-i", root_dir + "/video" + video_id + ".mp4", "tmp/frame%d.mp4"])
	with open("video" + video_id + "-timestamp.txt") as f:
		f.readline()
		i = 0
		for line in f:
			i += 1
			label = line.split('	')[1][:-1]
			if not destination_dir + "/video" + video_id + "/" in dirs:
				subprocess.call(["mkdir", destination_dir + "/video" + label])
				subprocess.call(["mkdir", destination_dir + "/video" + label + "/video" + video_id])
				subprocess.call(["mkdir", destination_dir + "/video" + video_id])
				subprocess.call(["mkdir", destination_dir + "/video" + video_id + "/"])
				dirs.add(destination_dir + "/video" + video_id + "/")

			subprocess.call(["mv", "tmp/frame" + str(i) + ".mp4", destination_dir + "/video" + video_id + "/"])



["10", "13", "19", "23", "22", "29", "32", "33", "38"]
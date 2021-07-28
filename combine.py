from moviepy.editor import *
import os
from natsort import natsorted

L =[]

for root, dirs, files in os.walk("our_data/combine/Drowning"):

    #files.sort()
    files = natsorted(files)
    for file in files:
        if os.path.splitext(file)[1] == '.avi':
            filePath = os.path.join(root, file)
            video = VideoFileClip(filePath)
            L.append(video)

final_clip = concatenate_videoclips(L)
final_clip.to_videofile("our_data/combine/drown.mp4", fps=30, remove_temp=False)   
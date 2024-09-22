import os
import yt_dlp
from random_sentence import random_sentence
from datetime import timedelta
from youtubesearchpython import VideosSearch
import re

def exist_video():
    return os.path.isfile("videos/video0.mp4")

def delete_video():
    for f in os.listdir("videos"):
        os.remove(os.path.join("videos", f))
            

def is_short_video(duration_str, duration_limit):
    parts = list(map(int, duration_str.split(':')))
    if len(parts) == 2:
        duration = timedelta(minutes = parts[0], seconds = parts[1])
    else:
        duration = timedelta(hours = parts[0], minutes = parts[1], seconds = parts[2])

    return duration < timedelta(seconds = duration_limit[1]) and duration > timedelta(seconds = duration_limit[0])

def configuration_at_time_step(time_step):
    if time_step < 1000:
        return ([5, 3 * 60], 64)
    elif time_step < 5000:
        return ([30, 5 * 60], 128)
    elif time_step < 10000:
        return ([60, 6 * 60], 128)
    elif time_step < 20000:
        return ([60, 6 * 60], 64)
    else:
        return ([2 * 60, 6 * 60], 32)

def download_video(time_step, training_phase):
    if not os.path.exists("videos"):
            os.mkdir("videos")

    if training_phase == "Autoencoder":
        for i in range(2):
            while True:
                try:
                    ytd = yt_dlp.YoutubeDL({
                        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                        "outtmpl": "videos/video" + str(i) + ".mp4"
                    })

                    is_search_done = False
                    v_id = None
                    while not is_search_done:
                        for v in VideosSearch(random_sentence(), 5).result()["result"]:
                            if is_short_video(v["duration"], [60, 3 * 60]):
                                is_search_done = True
                                v_id = v["id"]

                    ytd.download(["https://www.youtube.com/watch?v=" + v_id])
                    break
                except:
                    for f in os.listdir("videos"):
                        if f.startswith("video") and re.search(r"video(\d+)\.mp4", f) == None:
                            os.remove(os.path.join("videos", f))

    elif training_phase == "Stable Diffusion":
        duration, _ = configuration_at_time_step(time_step)

        while True:
            try:
                ytd = yt_dlp.YoutubeDL({
                    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "outtmpl": "videos/video0.mp4"
                })

                is_search_done = False
                v_id = None
                v_description = ""
                while not is_search_done:
                    for v in VideosSearch(random_sentence(), 5).result()["result"]:
                        if is_short_video(v["duration"], duration):
                            is_search_done = True
                            v_id = v["id"]
                            bonus_description = ""
                            if v["descriptionSnippet"] != None:
                                bonus_description = ". "
                                for desc in v["descriptionSnippet"]:
                                    bonus_description += desc["text"]

                            v_description = v["title"] + bonus_description

                ytd.download(["https://www.youtube.com/watch?v=" + v_id])
                with open("videos/description0.txt", "w") as f:
                    f.write(v_description)

                break
            except:
                for f in os.listdir("videos"):
                    if f.startswith("video") and re.search(r"video(\d+)\.mp4", f) == None:
                        os.remove(os.path.join("videos", f))
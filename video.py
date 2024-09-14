import os
import yt_dlp
from random_sentence import random_sentence
from datetime import timedelta
from youtubesearchpython import VideosSearch

def exist_video():
    return os.path.isfile("video.mp4")

def delete_video():
    return os.remove("video.mp4")

def is_less_than_10_min(duration_str):
    parts = list(map(int, duration_str.split(':')))
    if len(parts) == 2:
        duration = timedelta(minutes = parts[0], seconds = parts[1])
    else:
        duration = timedelta(hours = parts[0], minutes = parts[1], seconds = parts[2])

    return duration < timedelta(minutes = 10)

def download_video():
    ytd = yt_dlp.YoutubeDL({
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": "video.mp4"
    })

    is_search_done = False
    v_id = None
    while not is_search_done:
        for v in VideosSearch(random_sentence(), 5).result()["result"]:
            if is_less_than_10_min(v["duration"]):
                is_search_done = True
                v_id = v["id"]
                break

    ytd.download(["https://www.youtube.com/watch?v=" + v_id])


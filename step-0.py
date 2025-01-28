# scraper to scrape the metadata from rthk.hk podcast page

import os
import requests
import pandas as pd
from tqdm import tqdm
import random
from urllib.parse import quote
import xml.etree.ElementTree as ET
import multiprocessing
import argparse
import audioread
import librosa
import soundfile as sf

#
headers = {
    "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Chrome/56.0.2924.87",
    "Connection": "close",
}
# Example URL:
# https://podcast.rthk.hk/podcast/episodeList.php?pid=447&display=all&year=2020

# Example format
# <?xml version="1.0" encoding="utf-8"?>
# <episodeList>
# 	<total>306</total>
# 	<remainder>0</remainder>
# 	<page>0</page>
# 	<count>306</count>
# 	<episodePerPage>12</episodePerPage>
# <episode>
# 	<pid>447</pid>
# 	<eid>173854</eid>
# 	<episodeTitle><![CDATA[中國現代作家選集 蕭紅 (第二十六至三十集)]]></episodeTitle>
# 	<episodeDate>2020-12-31</episodeDate>
# 	<cover><![CDATA[https://podcast.rthk.hk/podcast/upload_photo/item_photo/16x9_275x155_447.jpg]]></cover>
# 	<duration>00:51:39</duration>
# 	<mediafile><![CDATA[https://podcasts.rthk.hk/podcast/media/audiobook/447_2101071456_35804.mp3]]></mediafile>
# 	<format>audio</format>
# </episode>
# ...
# </episodeList>
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

eids = {
    541: [
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
        2019,
        2018,
        2017,
        2016,
        2015,
        2014,
        2013,
    ],
    1843: [
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
    ],
    292: [2021, 2020],
    1223: [
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
        2019,
        2018,
        2017,
    ],
    42: [
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
        2019,
        2018,
        2017,
        2016,
        2015,
        2014,
        2013,
        2012,
        2011,
        2010,
        2009,
        2008,
        2007,
        2006,
        2005,
    ],
    825: [
        2025,
        2024,
        2023,
        2022,
        2021,
    ],
    836: [
        2025,
        2024,
    ],
    837: [
        2025,
        2024,
        2023,
        2022,
        2021,
    ],
    916: [
        2025,
        2024,
        2023,
        2022,
    ],
    914: [
        2025,
        2024,
        2023,
        2022,
    ],
    918: [
        2025,
        2024,
        2023,
        2022,
    ],
    919: [
        2025,
        2024,
        2023,
        2022,
    ],
    293: [  # 十萬八千里
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
        2019,
        2018,
    ],
    336: [
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
        2019,
        2018,
        2017,
        2016,
        2015,
        2014,
        2013,
        2012,
        2011,
    ],
    1886: [2021, 2022, 2023, 2024],
    447: [2021, 2020, 2019, 2018, 2017],
    23: [
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
        2019,
        2018,
        2017,
        2016,
        2015,
        2014,
        2013,
        2012,
        2011,
        2010,
        2009,
        2008,
        2007,
        2006,
        2005,
    ],
    833: [
        2025,
        2024,
        2023,
        2022,
        2021,
    ],
    902: [
        2025,
        2024,
        2023,
        2022,
        2021,
    ],
    662: [2017, 2016, 2015, 2014],
    207: [
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
        2019,
        2018,
        2017,
        2016,
        2015,
        2014,
        2013,
        2012,
    ],
    236: [
        2024,
        2023,
        2022,
        2021,
        2020,
        2019,
        2018,
        2017,
        2016,
        2015,
        2014,
        2013,
        2012,
        2011,
        2010,
    ],
    308: [
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
    ],
    289: [2025, 2024],
    307: [2025, 2024],
    1788: [2025, 2024],
    287: [
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
        2019,
        2018,
        2017,
        2016,
        2015,
        2014,
        2013,
        2012,
        2011,
    ],
    2121: [
        2025,
        2024,
        2023,
    ],
    356: [
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
        2019,
        2018,
        2017,
        2016,
        2015,
        2014,
        2013,
        2012,
    ],
    826: [
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
        2019,
        2018,
        2017,
        2016,
        2015,
        2014,
    ],
    1069: [
        2025,
        2024,
        2023,
        2022,
        2021,
        2020,
        2019,
        2018,
        2017,
        2016,
        2015,
        2014,
    ],
}
base_url = "https://podcast.rthk.hk/podcast/episodeList.php?pid={pid}&display=all"


def get_podcast_metadata(pid: int, year: int):
    url = f"{base_url.format(pid=str(pid))}&year={str(year)}"
    response = requests.get(url)
    root = ET.fromstring(response.text)
    episodes = root.findall("episode")
    metadata = []

    for episode in episodes:
        title = episode.find("episodeTitle").text
        eid = episode.find("eid").text
        pid = episode.find("pid").text
        date = episode.find("episodeDate").text
        cover = episode.find("cover").text
        duration = episode.find("duration").text
        mediafile = episode.find("mediafile").text
        metadata.append(
            {
                "eid": eid,
                "pid": pid,
                "title": title,
                "date": date,
                "cover": cover,
                "duration": duration,
                "mediafile": mediafile,
            }
        )

    return metadata


def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    attempt = 0

    while attempt < 3:
        try:
            # socks5_proxy = f"socks5://{NODRVPN_SERVICE_USERNAME}:{NODRVPN_SERVICE_PASSWORD}@{proxy_server}:1080"
            req = requests.get(
                url,
                headers=headers,
                timeout=10,
                # proxies={"https": http_proxy, "http": http_proxy},
            )
            with open(dst, "wb") as f:
                f.write(req.content)

        except Exception as e:
            print(f"Error: {e}")
            attempt += 1
        else:
            break


def convert_to_mono_16k_mp3(audio_file: str, out_file: str):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    wav, _ = librosa.load(audio_file, sr=16_000)
    sf.write(out_file, wav, 16_000, format="mp3")


def download_episode(episode):
    mediafile = episode["mediafile"]
    pid = episode["pid"]
    eid = episode["eid"]
    filename = mediafile.split("/")[-1]
    extension = os.path.splitext(filename)[1]
    out_file = f"audios/{pid}-{eid}{extension}"
    out_16k_file = out_file.replace("audios/", "audios_16k/").replace(extension, ".mp3")

    if os.path.exists(out_file):
        try:
            with audioread.audio_open(out_16k_file) as f:
                duration = f.duration
        except:
            duration = 0

        filesize = os.path.getsize(out_file)

        if filesize > 1000 and duration > 10:
            if os.path.exists(out_16k_file):
                return

            convert_to_mono_16k_mp3(out_file, out_16k_file)

            return

    try:
        download_from_url(mediafile, out_file)

        # if file size is too small(0.5mb), it is likely a 404 page
        if os.path.getsize(out_file) < 1000:
            os.remove(out_file)
            print(f"Cannot download: {mediafile}, the file is too small.")
            return

        # convert to 16k
        convert_to_mono_16k_mp3(out_file, out_16k_file)
    except:
        print(f"Cannot down: {mediafile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proc", type=int, default=2)
    args = parser.parse_args()

    if os.path.exists("metadata.csv"):
        metadata = pd.read_csv("metadata.csv").to_dict("records")
    else:
        metadata = []

        for pid, years in tqdm(eids.items()):
            for year in years:
                episodes = get_podcast_metadata(pid, year)
                metadata.extend(episodes)

    pd.DataFrame(metadata).to_csv("metadata.csv", index=False)

    # download mp3 files using multiprocessing
    with multiprocessing.Pool(processes=args.num_proc) as pool:
        mp3_files = list(
            tqdm(pool.imap(download_episode, metadata), total=len(metadata))
        )

    # without multiprocessing
    for episode in tqdm(metadata):
        download_episode(episode)

    print("All files downloaded.")

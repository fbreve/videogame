# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 12:20:57 2023

@author: fbrev
"""
import requests
import json
import pandas as pd
import time
import html

import credentials

# 28  - Atari 2600
# 22  - NES
# 26  - Master System
# 40  - PC Engine
# 16  - Mega Drive
# 15  - Super Nintendo
# 23  - Sega Saturn
# 6   - PlayStation
# 9   - Nintendo 64
# 8   - Dreamcast
# 7   - PlayStation 2
# 14  - GameCube
# 13  - Xbox
# 69  - Xbox 360
# 81  - PlayStation 3
# 82  - Wii
# 132 - Wii U
# 141 - PlayStation 4
# 142 - Xbox One
# 203 - Nintendo Switch
# 289 - Xbox Series
# 288 - PlayStation 5
# 143 - Arcade
platform = 143


offset = 0
moregames = True
gamelist = pd.DataFrame()

while(moregames):

    url = "https://api.mobygames.com/v1/games?api_key=" + credentials.api_key + "&format=brief&platform=" + str(platform) + "&offset=" + str(offset)
    
    print("Requesting games " + str(offset) + " to " + str(offset+100) + "...")

    while(True):
        try:
            response = requests.get(url)
        except requests.exceptions.RequestException as e:
            print("Error: " + str(e) + " Trying again...")
            time.sleep(3)
        else:
            break        
    
        if (response.status_code!=200):
            print("Something went wrong. API Status Code: " + str(response.status_code) + ". Trying Again...\n")
        else:
            break
                 
    json_data = json.loads(response.text)    

    if len(json_data["games"])>0: 
        df = pd.DataFrame.from_dict(json_data["games"])    
        gamelist = pd.concat([gamelist, df])
        offset = offset + 100
        time.sleep(10)
    if len(json_data["games"])<100:
        moregames=False
        
gamelist.title = gamelist.title.apply(html.unescape)
gamelist.reset_index(drop=True, inplace=True)
gamelist.to_csv("data\gamelist-" + str(platform) + ".csv")

print("Done.\n")
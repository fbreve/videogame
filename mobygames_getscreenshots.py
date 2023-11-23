# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:14:00 2023

@author: fbrev
"""
import pandas as pd
import requests
import json
import os
import time

import credentials

screenshots_path = "./data/screenshots/"

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

gamelist = pd.read_csv("./data/gamelist-" + str(platform) + ".csv", index_col=0)

gamecount = len(gamelist)

sslist = pd.DataFrame(columns=['game_id', 'filename'])
ssclist = pd.DataFrame(columns=['game_id', 'sc_count'])

for index, row in gamelist.iterrows():
           
    url = "https://api.mobygames.com/v1/games/" + str(row.game_id) + "/platforms/" + str(platform) + "/screenshots?api_key=" + credentials.api_key
    
    print("Requesting screenshots of game " + str(index+1) + " of " + str(gamecount) + " (id:" + str(row.game_id) + ") " + row.title + "... ")
       
    while(True):
        try:
            response = requests.get(url, timeout=10)
            if (response.status_code!=200):
                print("Something went wrong. API Status Code: " + str(response.status_code) + ". Trying again...")
                continue             
        except requests.exceptions.RequestException as e:
            print("Error: " + str(e) + " Trying again...")
            time.sleep(3)
        else:
            break
                
    start_time = time.time()
                   
    json_data = json.loads(response.text)   
    
    screenshotcount = len(json_data["screenshots"])
   
    df_sscrow = pd.DataFrame.from_dict({'game_id': [row.game_id], 'sc_count': [screenshotcount]})
    
    ssclist = pd.concat([ssclist, df_sscrow])
   
    if screenshotcount>0:
        
        print("Downloading " + str(screenshotcount) + " screenshots... ",end="")
        
        df = pd.DataFrame.from_dict(json_data["screenshots"])    
        
        for image in df.image:
            while(True):
                try:
                    screenshot = requests.get(image, allow_redirects=True, timeout=10)
                    if (screenshot.status_code!=200):
                        print("Something went wrong. API Status Code: " + str(response.status_code) + ". Trying again...")
                        continue
                except requests.exceptions.RequestException as e:
                    print("Error: " + str(e) + " Trying again...")
                    time.sleep(3)
                else:
                    break                                        
            filename = image.rsplit('/',1)[1]
            os.makedirs(os.path.dirname(screenshots_path + filename), exist_ok=True)
            open(screenshots_path + filename, 'wb').write(screenshot.content)          
            df_ssrow = pd.DataFrame.from_dict({'game_id': [row.game_id], 'filename': [filename]})                   
            sslist = pd.concat([sslist, df_ssrow])
        
    else:        
        
        print("There is no screenshots for this game. ", end="")        
        
    stop_time = time.time()
    time_elapsed = stop_time - start_time
   
    if (time_elapsed < 10):
        print("Finished in " + "{:4.2f}".format(time_elapsed) + " seconds. Waiting " + "{:4.2f}".format(10-time_elapsed) + " seconds...")
        time.sleep(10-time_elapsed)
    else:
        print("Finished in " + "{:4.2f}".format(time_elapsed) + " seconds.")
      
      
sslist.reset_index(drop=True, inplace=True)
ssclist.reset_index(drop=True, inplace=True)
sslist.to_csv("data\screenshotlist-" + str(platform) + ".csv")
ssclist.to_csv("data\screenshotcount-" + str(platform) + ".csv")

print("Done.\n")    
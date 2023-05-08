from bs4 import BeautifulSoup
import requests
import pandas as pd
import time


def get_shooting_info(match_data): #add type hints
    """
    match_data = match outcomes to join onto
    """

    print('Scraping shooting stats from')
    for l in BeautifulSoup(match_data.text).find_all('a'):

        # need to make sure the a-tag even has a href attribute, and filter by shooting links
        if l.get("href") and ('all_comps/shooting/' in l.get("href")):  
            shooting_data = requests.get(f"http://fbref.com{l.get('href')}")
            time.sleep(5) 
            print(f"http://fbref.com{l.get('href')}")
            shooting_df = pd.read_html(shooting_data.text, match='Shooting')[0] #again, returns more than one df for some reason
            shooting_df.columns = shooting_df.columns.droplevel()
            time.sleep(5)
            break #can have more than one of the same link, so break once its found for that team under consideration
     
    return shooting_df

def get_passing_info(match_data): #add type hints
    """
     match_data = match outcomes to join onto
    """

    print('Scraping passing stats from')
    for l in BeautifulSoup(match_data.text).find_all('a'):

        # need to make sure the a-tag even has a href attribute, and filter by shooting links
        if l.get("href") and ('all_comps/passing/' in l.get("href")):  
            passing_data = requests.get(f"http://fbref.com{l.get('href')}")
            time.sleep(5) 
            print(f"http://fbref.com{l.get('href')}")
            passing_df = pd.read_html(passing_data.text, match='Passing')[0] #again, returns more than one df for some reason
            passing_df.columns = passing_df.columns.droplevel()
            time.sleep(5)
            break #can have more than one of the same link, so break once its found for that team under consideration
     
    return passing_df

def get_possession_info(match_data): #add type hints
    """
     match_data = match outcomes to join onto
    """

    print('Scraping possesion stats from:')
    for l in BeautifulSoup(match_data.text).find_all('a'):

        # need to make sure the a-tag even has a href attribute, and filter by shooting links
        if l.get("href") and ('all_comps/possession/' in l.get("href")):  
            possession_data = requests.get(f"http://fbref.com{l.get('href')}")
            time.sleep(5) 
            print(f"http://fbref.com{l.get('href')}")
            possession_df = pd.read_html(possession_data.text, match='Possession')[0] #again, returns more than one df for some reason
            possession_df.columns = possession_df.columns.droplevel()
            time.sleep(5)
            break #can have more than one of the same link, so break once its found for that team under consideration

    return possession_df

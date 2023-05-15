from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import requests

def get_extra_info(
        match_data: requests.Response, 
        table_type: str
        ) -> pd.DataFrame: 

    """
    match_data = match outcomes to join onto
    table_type = name of table to join in to match data. Should have captial letter
    """

    print('Scraping shooting stats from')
    for l in BeautifulSoup(match_data.text, 'lxml').find_all('a'):

        # need to make sure the a-tag even has a href attribute, and filter by shooting links
        if l.get("href") and (f'all_comps/{table_type.lower()}/' in l.get("href")):  
            extra_data = requests.get(f"http://fbref.com{l.get('href')}")
            print(f"http://fbref.com{l.get('href')}")
            df = pd.read_html(extra_data.text, match=table_type)[0] #again, returns more than one df for some reason
            df.columns = df.columns.droplevel()
            time.sleep(7)
            break #can have more than one of the same link, so break once its found for that team under consideration

    return df


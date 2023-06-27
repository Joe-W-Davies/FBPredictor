from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import requests

def get_extra_info(
        match_data: requests.Response, 
        table_type: str,
        level_map: dict
        ) -> pd.DataFrame: 

    """
    Its not pretty now we have multi-level indexes, but it works...    

    match_data = match outcomes to join onto
    table_type = name of table to join in to match data. 
    """


    print(f'Scraping {table_type} stats from')
    for l in BeautifulSoup(match_data.text, 'lxml').find_all('a'):

        ###DEBUG
        #with open(f"data/html_debug_{table_type}.txt", "a+") as fn: #debug
        #    if (l.get("href") is not None):
        #        fn.write(l.get("href")) #debug
                #if (f'all_comps/{table_type.lower()}/' in l.get("href")):
                #    print(l.get('href'))
        ###DEBUG

        # need to make sure the a-tag even has a href attribute, and filter by shooting/possesion/etc links
        if l.get("href") and (f'all_comps/{table_type.lower()}/' in l.get("href")) and (l.get("href") is not None):
            print(f"http://fbref.com{l.get('href')}")
            extra_data = requests.get(f"http://fbref.com{l.get('href')}")
       
            with open(f"data/html_debug_{table_type}.txt", "w") as fn: #debug
                fn.write(extra_data.text) #debug

            #df = pd.read_html(extra_data.text, match=table_type)[0] 
            df = pd.read_html(extra_data.text)[0] 
            
            final_df = []
            col_names  = []
            for list_level_map in level_map:
                for t_level, bottom_list in list_level_map.items():
                    for b_level in bottom_list:
                        final_df.append(df[t_level][b_level])
                        col_names.append(t_level+b_level)


            #add date for joining later on
            df.columns = df.columns.droplevel()
            final_df.append(df['Date'])
            col_names.append('Date')

            final_df = pd.concat(final_df, axis=1)
            final_df.columns = col_names

            time.sleep(5)
            break #can have more than one of the same link, so break once its found for that team under consideration

    return final_df


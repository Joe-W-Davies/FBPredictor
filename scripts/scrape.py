from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import argparse
import yaml

from ScrapeUtils import get_possession_info, get_shooting_info, get_passing_info


#scrape 
def main(options):
    #unpack config

    with open(options.config, 'r') as config_file:
        config = yaml.load(config_file)
        years = config['years']
        league_links = config['league_links']
        match_vars = config['match_vars']

        extra_tables_to_scrape = config['extra_tables_to_scrape']
        possession_vars = config['possession_vars']
        passing_vars = config['passing_vars']
        shooting_vars = config['shooting_vars']



    for league_name,web_page in league_links.items():
    
        total_df = []
        for year in years:
        
            print(f'scraping data from: {year}')
            data = requests.get(web_page)
            soup = BeautifulSoup(data.text)
            
            # filter HTML to to get the main standings table (syntax is: html_tag.class)
            league_table = soup.select('table.stats_table')[0] #NOTE: why does this return multiple entries?
            
            team_tags = league_table.find_all('a')
           
            team_urls = [f"http://fbref.com{l.get('href')}" for l in team_tags if 'squads' in l.get('href')]
        
            for tu in team_urls:
                team_name =  tu.split('/')[-1].replace('-Stats','')
                print(f"\nScraping {team_name} Stats")
                match_data = requests.get(tu) 
                time.sleep(5) 
                
                print('Scraping match outcomes data from:')
                print(tu)
                match_info = pd.read_html(match_data.text, match='Scores & Fixtures')
                team_data_df = match_info[0] #can return more than 1 table for some reason
                team_data_df = team_data_df[match_vars] # filter match df
        
        
                    
                #get additional stats for a single team
                #FIXME: you can make the scapers into one function and just change e.g. all "Passing"->"Shooting"
                #Can also write a yaml linking possesion vars -> [var1, var2, var3, ...]
        
                if 'possession' in extra_tables_to_scrape:
                    original_rows = team_data_df.shape[0] 
                    possession_df = get_possession_info(match_data=match_data)
    
                    #FIXME: would be nice to join replace with a function since repetitive, but dunno how to return a continue if we got an error!!
                    #Date will be a unique ID for that team since can't play 2 matches in the same day
                    try:
                        team_data_df = team_data_df.merge(possession_df[possession_vars], on="Date", how='inner')
                    except ValueError: #sometimes shooting data doesn't exist
                        continue
                    lost_rows = original_rows - team_data_df.shape[0]
                    print(f'Inner joining match result + possesion tables resulted in losing {lost_rows} rows') # will likely be the case for current league where matches are still to be played
    
        
                if 'passing' in extra_tables_to_scrape:
                    original_rows = team_data_df.shape[0] 
                    passing_df = get_passing_info(match_data=match_data)
    
                    try:
                        team_data_df = team_data_df.merge(passing_df[passing_vars], on="Date", how='inner')
                    except ValueError: 
                        continue
                    lost_rows = original_rows - team_data_df.shape[0]
                    print(f'Inner joining match result + passing tables resulted in losing {lost_rows} rows') # will likely be the case for current league where matches are still to be played
        
        
                if 'shooting' in extra_tables_to_scrape:
                    original_rows = team_data_df.shape[0] 
                    shooting_df = get_shooting_info(match_data=match_data)
        
                    try:
                        team_data_df = team_data_df.merge(shooting_df[shooting_vars], on="Date", how='inner')
                    except ValueError: 
                        continue
        
    
                    lost_rows = original_rows - team_data_df.shape[0]
                    print(f'Inner joining match result + shooting tables resulted in losing {lost_rows} rows') # will likely be the case for current league where matches are still to be played
              
                #add some meta data
                team_data_df['team'] = team_name
                team_data_df['year'] = year
                total_df.append(team_data_df)
    
            #go back to previous year url
            previous_season = soup.select("a.prev")[0].get("href")
            web_page = f"https://fbref.com{previous_season}"
        
        total_df = pd.concat(total_df)
        total_df.columns = [c.lower() for c in total_df.columns]
        print(total_df.shape)

        with open(f'data/match_data_{league_name}.csv', 'w+', encoding = 'utf-8-sig') as f:
            total_df.to_csv(f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    options=parser.parse_args()
    main(options)

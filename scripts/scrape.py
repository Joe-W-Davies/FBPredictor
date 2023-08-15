from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import argparse
import yaml

from ScrapeUtils import get_extra_info


#scrape 
def main(options):

    with open(options.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
        years = config['years']
        league_links = config['league_links']
        match_vars = config['match_vars']
        extra_tables_name_to_var = config['extra_tables_to_scrape']



    for league_name,web_page in league_links.items():
    
        total_df = []
        for year in years:
        
            print(f'Scraping data from: {str(year)}-{str(year+1)}')
            web_page_year = web_page.replace('!YEAR!',f'{str(year)}-{str(year+1)}')
            data = requests.get(web_page_year)
            soup = BeautifulSoup(data.text, 'lxml')
            
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
                match_info = pd.read_html(
                    match_data.text, 
                    match='Scores & Fixtures'
                )
                #can return more than 1 table for some reason so get [0]
                team_data_df = match_info[0] 
                team_data_df = team_data_df[match_vars] 
        
                    
                #get additional stats for a single team
                #Can also write a yaml linking possesion vars -> [var1, var2, var3, ...]
                for extra_table_name, extra_table_map in extra_tables_name_to_var.items():
                    extra_df = get_extra_info(
                        match_data=match_data, 
                        table_type=extra_table_name,
                        level_map=extra_table_map
                    )

    
                    #Date will be a unique ID for that team since can't play 2 matches in the same day
                    try:
                        team_data_df = team_data_df.merge(
                            extra_df, 
                            on="Date", 
                            how='left'
                            )
                    except ValueError: #sometimes shooting data doesn't exist
                        continue
        
                #add some meta data
                team_data_df['team'] = team_name
                team_data_df['year'] = year
                total_df.append(team_data_df)

            with open(
                    f'data/per_year/match_data_{league_name}_more_vars_year_{year}.csv', 
                    'w+', 
                    encoding = 'utf-8-sig'
            ) as f:
                yr_df = pd.concat(total_df)
                yr_df.columns = [c.lower() for c in yr_df.columns]
                yr_df = yr_df.query(f'year=={year}')
                yr_df.to_csv(f)
    
            #go back to previous year url
            previous_season = soup.select("a.prev")[0].get("href")
            #web_page = f"https://fbref.com{previous_season}"
        
        total_df = pd.concat(total_df)
        total_df.columns = [c.lower() for c in total_df.columns]

        with open(
                f'data/leagues/match_data_{league_name}_more_vars.csv', 
                'w+', 
                encoding = 'utf-8-sig'
        ) as f:
            total_df.to_csv(f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    options=parser.parse_args()
    main(options)

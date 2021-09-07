# from nba_api.stats.endpoints import playbyplay
import Codes
import pandas as pd
import os

PROCESSED_PBP_PATH = r'D:\NBA_Project\data\play by play\processed'
RAW_PBP_PATH = r'D:\NBA_Project\data\play by play\raw'

class PBP_Processor():

    def __init__(self, using_cluster):
        self.using_cluster = using_cluster

    def preprocess(self, file_path, game_id=None):
        # Merge misses and rebounds
        # Delete free throws, subs (enters the game)
        # Merge events with the same time
        if game_id:

            if len(game_id) < 10:
                game_id = ('0' * (10 - len(game_id))) + str(game_id)

        elif file_path:
            df = pd.read_csv(file_path)

        df = self.filter_unnecessary_events(df)
        df = self.merge_misses_and_rebounds(df)
        df = self.set_pbp_start_time(df)
        df = self.filter_same_time_events(df)

        df = df[['description', 'extra_description', 'And1', 'quarter', 'start_time',
                      'end_time', 'team', 'home_score', 'away_score', 'attacking_team']] # Reorder columns
        df['duration'] = [row['start_time'] - row['end_time'] for i, row in df.iterrows()]
        df = self.format_description(df)
        df = df[df['duration'] >= Codes.POSSESSION_MIN_LENGTH]
        return df

    @staticmethod
    def modify_api_data(df):
        df['description'] = df[['HOMEDESCRIPTION', 'NEUTRALDESCRIPTION', 'VISITORDESCRIPTION']].agg(' '.join, axis=1)
        df = df.drop(['HOMEDESCRIPTION', 'NEUTRALDESCRIPTION', 'VISITORDESCRIPTION'], axis=1)
        df['home_score'] = [int(score[:score.find('-')]) for score in df['SCORE']]
        df['away_score'] = [int(score[:score.find('-') + 1:]) for score in df['SCORE']]

        df.rename(columns={'PERIOD': 'quarter', 'PCTIMESTRING': 'start_time', })

    def merge_misses_and_rebounds(self, pbp_frame):
        # Add rebound column:
        pbp_frame['rebound'] = None

        rows = list()
        merge_flag = False
        miss_description = ''
        extra_description = ''
        attacking_team = None
        for i, play in pbp_frame.iterrows():

            if 'miss' in play['description']:
                merge_flag = True
                miss_description = play['description']
                attacking_team = play[3]
            else:
                description = play[8]
                if merge_flag:
                    extra_description = description
                    description = miss_description
                    merge_flag = False

                if attacking_team is None:
                    attacking_team = play[3]

                play_dict = {'description': description.lower(),
                             'extra_description': extra_description.lower(),
                             'attacking_team': attacking_team,
                             'quarter': play[0],
                             'end_time': play[2],
                             'team': play[3],
                             'home_score': play[7],
                             'away_score': play[6]}

                attacking_team = None
                rows.append(play_dict)
                extra_description = ''
        plays = pd.DataFrame(rows)

        return plays

    def filter_unnecessary_events(self, pbp_frame):
        pbp_frame = pbp_frame[~(pbp_frame['description'].str.contains('free') |
                                pbp_frame['description'].str.contains('enters'))]
        return pbp_frame

    def set_pbp_start_time(self, plays_df):
        plays_df['start_time'] = plays_df['end_time'].shift(1)
        for i, row in plays_df.iterrows():
            if i == 0 or row['quarter'] != plays_df.iloc[i - 1]['quarter']:
                plays_df.at[i, 'start_time'] = 720
        return plays_df

    def filter_same_time_events(self, plays_df):
        plays_df['And1'] = 0
        plays_df['time_difference'] = plays_df['start_time'] - plays_df['end_time']
        drop_rows = []
        for i, row in plays_df.iterrows():
            if row['time_difference'] == 0:
                drop_rows.append(i)
                # Check if its AND1
                if ('makes' in row['description'] and 'Shooting foul' in plays_df.iloc[i - 1]['description']) or \
                    ('Shooting foul' in row['description'] and 'makes' in plays_df.iloc[i - 1]['description']):
                    plays_df.at[i - 1, 'extra_description'] = row['description']
                    plays_df.at[i - 1, 'And1'] = 1
        plays_df = plays_df.drop(drop_rows)

        return plays_df

    def format_description(self, plays_df):
        plays_df = self.add_columns(plays_df)

        for i, row in plays_df.iterrows():
            description = row['description']
            extra_description = row['extra_description']
            if 'miss' in description:
                plays_df.at[i, 'miss'] = 1
            if 'make' in description:
                plays_df.at[i, 'make'] = 1
            if '2pt' in description or '2-pt' in description:
                plays_df.at[i, '2pt'] = 1
            if '3pt' in description or '3-pt' in description:
                plays_df.at[i, '3pt'] = 1
            if 'shooting foul' in description or 'shooting foul' in extra_description:
                plays_df.at[i, 'shooting_foul'] = 1
            elif 'foul' in description:
                plays_df.at[i, 'foul'] = 1
            if 'Turnover' in description:
                plays_df.at[i, 'turnover'] = 1
            if 'bad pass' in description:
                plays_df.at[i, 'bad pass'] = 1
            if 'Violation' in description:
                plays_df.at[i, 'violation'] = 1
            if 'timeout' in description:
                plays_df.at[i, 'timeout'] = 1
            if 'Defensive rebound' in extra_description:
                plays_df.at[i, 'defensive_rebound'] = 1
            if 'Offensive rebound' in extra_description:
                plays_df.at[i, 'offensive_rebound'] = 1
            if 'REBOUND' in extra_description:
                plays_df.at[i, 'rebound'] = 1
            if 'block' in description:
                plays_df.at[i, 'block'] = 1

            description = description.split()
            for j, word in enumerate(description):
                if '.' in word:
                    plays_df.at[i, 'player_name'] = word + description[j + 1]
                    break
        return plays_df

    def add_columns(self, df):

        df['player_name'] = ''
        new_columns = ['miss', 'make', '2pt', '3pt', 'foul', 'shooting_foul', 'turnover', 'violation', 'bad pass',
                       'timeout', 'defensive_rebound', 'offensive_rebound', 'rebound', 'block']
        new_values = [0] * len(new_columns)
        df = df.reindex(columns=df.columns.tolist() + new_columns)  # add empty cols
        df[new_columns] = new_values  # multi-column assignment works for existing cols

        return df

    def pbp_stats(self, pbp_dir_path):
        plays_count = 0
        games_count = 0
        for filename in os.listdir(pbp_dir_path):
            if '7z' not in filename:
                print(filename)
                df = pd.DataFrame.from_csv(pbp_dir_path + '\\' + filename)
                plays_count += len(df)
                games_count += 1

        print(plays_count / games_count)
        print(games_count)

    def add_player_id(self, file_path, players_dict, alternative_names):
        unidentified_players = set()
        df = pd.DataFrame.from_csv(file_path)
        df['player_id'] = None
        for i, row in df.iterrows():
            player_name = row['player_name']
            if player_name is not None:
                if player_name in players_dict:
                    df.at[i, 'player_id'] = players_dict[player_name]
                elif player_name in alternative_names:
                    alternate_name = alternative_names[player_name]
                    df.at[i, 'player_id'] = players_dict[alternate_name]
                else:
                    unidentified_players.add(player_name)
        return df, unidentified_players

    def convert_players_index_to_dict(self, players_index_path):
        players_index = pd.read_csv(players_index_path, encoding='utf-8')
        dict = {}
        for i, row in players_index.iterrows():
            dict[row['pbp_name']] = row['playerid']
        return dict

    def add_player_ids_to_all(self, folder_path, players_dict):
        unidentified_players = set()
        for file in os.listdir(folder_path):
            updated_df, players = self.add_player_id(folder_path + '\\' + file, players_dict)
            updated_df.to_csv(folder_path + '\\' + file[:-4] + '_updated.csv')
            unidentified_players = unidentified_players|players
        return unidentified_players

    def fit_similar_names(self, players_index, names_list):
        names_dictionary = {}
        for name in names_list:
            for i in range(1, 4):
                sub_name = name[:-i]
                bol_names = players_index['pbp_name'].str.contains(sub_name, na=False, regex=False)
                if bol_names.any():
                    player_pbp_name = players_index[bol_names].iloc[0]['pbp_name']
                    names_dictionary[name] = player_pbp_name
                    break
        return names_dictionary

import pandas as pd
import py7zr
import os

# Thresholds:
PASSING_LANE_MIN_DIS = 3.5
SHOT_HEIGHT_THRESHOLD = 8.5 # 9 Feet is the basket height
POSSESSION_THRESHOLD_FEET = 3 # Approx 1 meter, as mentioned at: https://squared2020.com/2017/05/07/identifying-player-possession-in-spatio-temporal-data/
POSSESSION_MIN_LENGTH = 3
sequence_length = 16
X_MID = 47
Y_MID = 25
X_MAX = 94
Y_MAX = 50

PASS_TYPES = ['pass', 'handoff', 'bad pass']

BALL_COLOR = '#ff8c00'

using_cluster = False

if using_cluster:
    players_dict = pd.read_csv(r'/home/chenyan/NBA/Files/players.csv')
    season_stats = pd.read_csv(r'/home/chenyan/NBA/Files/Season Stats.csv')
else:
    players_dict = pd.read_csv(r'C:\Chen\NBA Project\NBA_Tracking_Project\Files\players.csv')
    season_stats = pd.read_csv(r'C:\Chen\NBA Project\NBA_Tracking_Project\Files\Season Stats.csv')

TEAM_COLORS_DICT = {
    1610612737: ('#E13A3E', 'ATL'),
    1610612738: ('#008348', 'BOS'),
    1610612751: ('#061922', 'BKN'),
    1610612766: ('#1D1160', 'CHA'),
    1610612741: ('#CE1141', 'CHI'),
    1610612739: ('#860038', 'CLE'),
    1610612742: ('#007DC5', 'DAL'),
    1610612743: ('#4D90CD', 'DEN'),
    1610612765: ('#006BB6', 'DET'),
    1610612744: ('#FDB927', 'GSW'),
    1610612745: ('#CE1141', 'HOU'),
    1610612754: ('#00275D', 'IND'),
    1610612746: ('#ED174C', 'LAC'),
    1610612747: ('#552582', 'LAL'),
    1610612763: ('#0F586C', 'MEM'),
    1610612748: ('#98002E', 'MIA'),
    1610612749: ('#00471B', 'MIL'),
    1610612750: ('#005083', 'MIN'),
    1610612740: ('#002B5C', 'NOP'),
    1610612752: ('#006BB6', 'NYK'),
    1610612760: ('#007DC3', 'OKC'),
    1610612753: ('#007DC5', 'ORL'),
    1610612755: ('#006BB6', 'PHI'),
    1610612756: ('#1D1160', 'PHX'),
    1610612757: ('#E03A3E', 'POR'),
    1610612758: ('#724C9F', 'SAC'),
    1610612759: ('#BAC3C9', 'SAS'),
    1610612761: ('#CE1141', 'TOR'),
    1610612762: ('#00471B', 'UTA'),
    1610612764: ('#002B5C', 'WAS'),
}

TEAM_IDS = {
    'ATLANTA HAWKS': '1610612737',
    'BOSTON CELTICS': '1610612738',
    'BROOKLYN NETS': '1610612751',
    'CHARLOTTE HORNETS': '1610612766',
    'CHICAGO BULLS': '1610612741',
    'CLEVELAND CAVALIERS': '1610612739',
    'DALLAS MAVERICKS': '1610612742',
    'DENVER NUGGETS': '1610612743',
    'DETROIT PISTONS': '1610612765',
    'GOLDEN STATE WARRIORS': '1610612744',
    'HOUSTON ROCKETS': '1610612745',
    'INDIANA PACERS': '1610612754',
    'LA CLIPPERS': '1610612746',
    'LOS ANGELES CLIPPERS': '1610612746',
    'LOS ANGELES LAKERS': '1610612747',
    'MEMPHIS GRIZZLIES': '1610612763',
    'MIAMI HEAT': '1610612748',
    'MILWAUKEE BUCKS': '1610612749',
    'MINNESOTA TIMBERWOLVES': '1610612750',
    'NEW ORLEANS PELICANS': '1610612740',
    'NEW YORK KNICKS': '1610612752',
    'OKLAHOMA CITY THUNDER': '1610612760',
    'ORLANDO MAGIC': '1610612753',
    'PHILADELPHIA 76ERS': '1610612755',
    'PHOENIX SUNS': '1610612756',
    'PORTLAND TRAIL BLAZERS': '1610612757',
    'SACRAMENTO KINGS': '1610612758',
    'SAN ANTONIO SPURS': '1610612759',
    'TORONTO RAPTORS': '1610612761',
    'UTAH JAZZ': '1610612762',
    'WASHINGTON WIZARDS': '1610612764',
}

Embed_Team_IDs = {
    'ATLANTA HAWKS': 0,
    'BOSTON CELTICS': 1,
    'BROOKLYN NETS': 3,
    'CHARLOTTE HORNETS': 4,
    'CHICAGO BULLS': 5,
    'CLEVELAND CAVALIERS': 6,
    'DALLAS MAVERICKS': 7,
    'DENVER NUGGETS': 8,
    'DETROIT PISTONS': 9,
    'GOLDEN STATE WARRIORS': 10,
    'HOUSTON ROCKETS': 11,
    'INDIANA PACERS': 12,
    'LA CLIPPERS': 13,
    'LOS ANGELES CLIPPERS': 13,
    'LOS ANGELES LAKERS': 14,
    'MEMPHIS GRIZZLIES': 15,
    'MIAMI HEAT': 16,
    'MILWAUKEE BUCKS': 17,
    'MINNESOTA TIMBERWOLVES': 20,
    'NEW ORLEANS PELICANS': 21,
    'NEW YORK KNICKS': 22,
    'OKLAHOMA CITY THUNDER': 23,
    'ORLANDO MAGIC': 24,
    'PHILADELPHIA 76ERS': 25,
    'PHOENIX SUNS': 26,
    'PORTLAND TRAIL BLAZERS': 27,
    'SACRAMENTO KINGS': 28,
    'SAN ANTONIO SPURS': 29,
    'TORONTO RAPTORS': 30,
    'UTAH JAZZ': 31,
    'WASHINGTON WIZARDS': 32,
}

Embed_IDs_To_Names = {
    0: 'ATLANTA HAWKS',
    1: 'BOSTON CELTICS',
    3: 'BROOKLYN NETS',
    4: 'CHARLOTTE HORNETS',
    5: 'CHICAGO BULLS',
    6: 'CLEVELAND CAVALIERS',
    7: 'DALLAS MAVERICKS',
    8: 'DENVER NUGGETS',
    9: 'DETROIT PISTONS',
    10: 'GOLDEN STATE WARRIORS',
    11: 'HOUSTON ROCKETS',
    12: 'INDIANA PACERS',
    # 13: 'LA CLIPPERS',
    13: 'LOS ANGELES CLIPPERS',
    14: 'LOS ANGELES LAKERS',
    15: 'MEMPHIS GRIZZLIES',
    16: 'MIAMI HEAT',
    17: 'MILWAUKEE BUCKS',
    20: 'MINNESOTA TIMBERWOLVES',
    21: 'NEW ORLEANS PELICANS',
    22: 'NEW YORK KNICKS',
    23: 'OKLAHOMA CITY THUNDER',
    24: 'ORLANDO MAGIC',
    25: 'PHILADELPHIA 76ERS',
    26: 'PHOENIX SUNS',
    27: 'PORTLAND TRAIL BLAZERS',
    28: 'SACRAMENTO KINGS',
    29: 'SAN ANTONIO SPURS',
    30: 'TORONTO RAPTORS',
    31: 'UTAH JAZZ',
    32: 'WASHINGTON WIZARDS',
}

team_code_to_shortcut = {
    1610612737: 'ATL',
    1610612738: 'BOS',
    1610612751: 'BKN',
    1610612766: 'CHA',
    1610612741: 'CHI',
    1610612739: 'CLE',
    1610612742: 'DAL',
    1610612743: 'DEN',
    1610612765: 'DET',
    1610612744: 'GSW',
    1610612745: 'HOU',
    1610612754: 'IND',
    1610612746: 'LAC',
    1610612747: 'LAL',
    1610612763: 'MEM',
    1610612748: 'MIA',
    1610612749: 'MIL',
    1610612750: 'MIN',
    1610612740: 'NOP',
    1610612752: 'NYK',
    1610612760: 'OKC',
    1610612753: 'ORL',
    1610612755: 'PHI',
    1610612756: 'PHX',
    1610612757: 'POR',
    1610612758: 'SAC',
    1610612759: 'SAS',
    1610612761: 'TOR',
    1610612762: 'UTA',
    1610612764: 'WAS',
}

TEAM_CODES = {
    1610612737: 'ATLANTA HAWKS',
    1610612738: 'BOSTON CELTICS',
    1610612751: 'BROOKLYN NETS',
    1610612766: 'CHARLOTTE HORNETS',
    1610612741: 'CHICAGO BULLS',
    1610612739: 'CLEVELAND CAVALIERS',
    1610612742: 'DALLAS MAVERICKS',
    1610612743: 'DENVER NUGGETS',
    1610612765: 'DETROIT PISTONS',
    1610612744: 'GOLDEN STATE WARRIORS',
    1610612745: 'HOUSTON ROCKETS',
    1610612754: 'INDIANA PACERS',
    # 1610612746: 'LA CLIPPERS',
    1610612746: 'LOS ANGELES CLIPPERS',
    1610612747: 'LOS ANGELES LAKERS',
    1610612763: 'MEMPHIS GRIZZLIES',
    1610612748: 'MIAMI HEAT',
    1610612749: 'MILWAUKEE BUCKS',
    1610612750: 'MINNESOTA TIMBERWOLVES',
    1610612740: 'NEW ORLEANS PELICANS',
    1610612752: 'NEW YORK KNICKS',
    1610612760: 'OKLAHOMA CITY THUNDER',
    1610612753: 'ORLANDO MAGIC',
    1610612755: 'PHILADELPHIA 76ERS',
    1610612756: 'PHOENIX SUNS',
    1610612757: 'PORTLAND TRAIL BLAZERS',
    1610612758: 'SACRAMENTO KINGS',
    1610612759: 'SAN ANTONIO SPURS',
    1610612761: 'TORONTO RAPTORS',
    1610612762: 'UTAH JAZZ',
    1610612764: 'WASHINGTON WIZARDS'
}

def create_player_to_id_file(data_folder_path):
    teams = {}
    for filename in os.listdir(data_folder_path):
        # if len(teams.keys()) == 2:
        #     break
        if filename.endswith('7z'):
            print(filename)
            json_name = None
            with py7zr.SevenZipFile(data_folder_path + '\\' + filename, 'r') as archive:
                allfiles = archive.getnames()
                if allfiles:
                    json_name = allfiles[0]
                    if not os.path.exists(data_folder_path + '\\' + json_name[:-5] + '.csv'):
                        archive.extractall(data_folder_path) # Extract archive in folder

            if json_name is not None and os.path.exists(data_folder_path + '\\' + json_name):
                visitor, home = get_player_dict(data_folder_path + '\\' + json_name)
                if visitor['name'] not in teams.keys():
                    teams[visitor['name']] = {}
                if home['name'] not in teams.keys():
                    teams[home['name']] = {}
                # teams[visitor['name']] = visitor['players']
                # teams[home['name']] = home['players']

                visitor_players = create_dict_from_players_list(visitor['players'])
                home_players = create_dict_from_players_list(home['players'])

                teams[visitor['name']] = {**teams[visitor['name']], **visitor_players}
                teams[home['name']] = {**teams[home['name']], **home_players}

                os.remove(data_folder_path + '\\' + json_name)

    frames = []
    for key, value in teams.items():
        team_df = pd.DataFrame(list(value.values()))
        team_df['team'] = key
        frames.append(team_df)
    players_index = pd.concat(frames)
    players_index['pbp_name'] = [player['firstname'][0] + '.' + player['lastname'] for i, player in
                                 players_index.iterrows()]
    players_index.to_csv('players.csv')

def create_dict_from_players_list(players):
    d = dict()
    for player in players:
        d[player['playerid']] = player
    return d

def get_player_dict(file_path):
    df = pd.read_json(file_path)
    event = df.iloc[0]['events']
    visitor = event['visitor']
    home = event['home']
    return visitor, home

def get_player_name_by_id(player_id):
    player_data = players_dict.loc[players_dict['playerid'] == player_id]
    if player_data is None:
        return None
    else:
        return player_data['pbp_name'].iat[0]

def get_player_jersey_num_by_id(player_id):
    player_data = players_dict.loc[players_dict['playerid'] == player_id]
    if player_data is None:
        return None
    else:
        return player_data['jersey'].iat[0]

def get_player_embed_id(player_id):
    player_data = players_dict.loc[players_dict['playerid'] == player_id]
    if player_data is None:
        return None
    else:
        try:
            return player_data['embed_player_id'].iat[0]
        except:
            return None

def get_player_name_from_embed_id(embed_id):
    player_data = players_dict.loc[players_dict['embed_player_id'] == embed_id]
    if player_data is None:
        return None
    else:
        try:
            return player_data['pbp_name'].iat[0]
        except:
            return None

def get_locations_df_column_names():
    locations_df_column_names = [['ball_x', 'ball_y', 'ball_z']] + \
                                [['player_' + str(i) + '_' + coor for coor in ['x', 'y']] for i in range(10)]
    flatten = lambda l: [item for sublist in l for item in sublist]
    locations_df_column_names = flatten(locations_df_column_names)
    return locations_df_column_names

def get_player_stats_by_id(player_id, defending=False):
    player_data = players_dict.loc[players_dict.playerid == player_id]
    cols = ['height_norm_ga', 'weight_norm_ga', 'PF', 'SG', 'SF', 'PF', 'C']
    if not defending:
        cols += ['FT%', '2P%', '3P%']
    return player_data[cols].values[0].tolist()

# create_player_to_id_file(r'C:\Chen\NBA Project\Raw Data\data\2016.NBA.Raw.SportVU.Game.Logs')
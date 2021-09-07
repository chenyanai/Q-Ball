import sys
sys.path.append('/home/chenyan/NBA')

from Data_Loader_Converter import Data_Loader
import multiprocessing as mp
import Codes

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    offense_model = True
    separate_actor_critic = True
    dueling = False
    create_new_episodes = True
    train_just_critic = False

    sequence_length = Codes.sequence_length
    state_dim = 32 + 1  # + 1 for the is_attacking_the_left feature

    attacking_players_stats = 50
    defending_players_stats = 35

    discrete_action_dim = 6
    continuous_action_dim = 13

    if not offense_model:
        state_dim = state_dim + discrete_action_dim

    if Codes.using_cluster:
        models_folder = r'/home/chenyan/NBA/Models/DQN_models/'
        data_path = r'/home/chenyan/NBA/data/processed pairs V2'
    else:
        models_folder = r'C:\Chen\NBA Project\NBA_Tracking_Project\Models\DQN_models'
        data_path = r'D:\NBA_Project\data\processed pairs gameV2'

    mp.set_start_method('spawn')
    pool = mp.Pool(processes=mp.cpu_count())

    data_loader = Data_Loader(data_path, sequence_length=sequence_length, load_games_num=1000, transpose=False,
                                   skip_frames=6, create_new_episodes_data=create_new_episodes, offense=offense_model,
                                   pool=pool)

from Preprocessing.Play_by_Play_Preprocessing import PBP_Processor
from Preprocessing.Moments_Preprocessing import Moments_Processor
from DDPG_Model.EpisodesCreation import Data_Loader
from Codes import using_cluster
from Game import Game

import multiprocessing as mp
import Codes
import os


import warnings
warnings.filterwarnings("ignore")

def process_game_data(file, output_path, m_processor, pbp_processor, pbp_dir):
    print(f'Starting to work on {file}')
    pbp_path = os.path.join(pbp_dir, file + '.csv')
    moments_path = os.path.join(moments_dir, file)
    game = Game(moments_json_path=moments_path,
                pbp_file_path=pbp_path,
                moments_processor=m_processor,
                pbp_processor=pbp_processor,
                using_cluster=using_cluster,
                output_path=output_path)
    game.save_game_as_jsons(output_path)
    print(f'Finished working on {file}')

if __name__ == '__main__':

    pbp_dir = r'data\raw_data\play_by_play'
    moments_dir = r'data\raw_data\moments'
    output_path = r'data\processed_data\merged_games'
    episodes_path = r'data\processed_data\episodes'

    create_new_episodes = True

    sequence_length = Codes.sequence_length
    state_dim = 32 + 1  # + 1 for the is_attacking_the_left feature

    attacking_players_stats = 50
    defending_players_stats = 35

    discrete_action_dim = 6
    continuous_action_dim = 13

    # Step 1: Merging play by play and moments data sets
    m_processor = Moments_Processor(using_cluster=using_cluster)
    pbp_processor = PBP_Processor(using_cluster=using_cluster)

    for file in os.listdir(moments_dir):
        process_game_data(file, output_path, m_processor, pbp_processor, pbp_dir)

    # Step 2: Episodes creation
    input_data_path = output_path
    episodes_data_path = r'data\processed_data\episodes'
    models_folder = r'models'

    mp.set_start_method('spawn')
    pool = mp.Pool(processes=mp.cpu_count())

    data_loader = Data_Loader(
        input_data_path,
      episodes_save_path=episodes_path,
      sequence_length=sequence_length,
      load_games_num=1000,
      transpose=True,
      skip_frames=6,
      create_new_episodes_data=create_new_episodes,
      pool=pool
    )
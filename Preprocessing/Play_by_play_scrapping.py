from basketball_reference_web_scraper.data import TEAM_ABBREVIATIONS_TO_TEAM, OutputType, OutputWriteOption
from basketball_reference_web_scraper import client
import os

if __name__ == '__main__':
    moments_dir = r'data\raw_data\moments'
    file = ''
    for file in os.listdir(moments_dir):
        pbp_file_path = os.path.join(r'data\raw_data\play_by_play', file + '.csv')

        if not os.path.isfile(pbp_file_path):
            month = file[:2]
            day = file[3:5]
            year = file[6:10]
            away_team = file[11:14]
            home_team = file[18:21]

            if away_team == 'BKN':
                away_team = 'BRK'
            if home_team == 'BKN':
                home_team = 'BRK'

            if away_team == 'PHX':
                away_team = 'PHO'
            if home_team == 'PHX':
                home_team = 'PHO'

            play_by_play = client.play_by_play(
                home_team=TEAM_ABBREVIATIONS_TO_TEAM[home_team],
                year=int(year),
                month=int(month),
                day=int(day),
                output_type=OutputType.CSV,
                output_write_option=OutputWriteOption.CREATE_AND_WRITE,
                output_file_path=pbp_file_path
            )

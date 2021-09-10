from Data_Loader_Converter import Data_Loader

from DDPG_Model.Actor_Critic import ActorCriticNetwork
from DDPG_Model.Critic import CriticNetwork
from DDPG_Model.Actor import ActorNetwork
from DDPG_Model.Buffer import Buffer

from tensorflow.keras.models import load_model
from datetime import datetime
import tensorflow as tf
import Codes
import json
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def save_models(experiment_dir, episode_num, done=False):
    if done:
        for file in os.listdir(experiment_dir):
            if 'parameters' not in file:
                os.remove(os.path.join(experiment_dir, file))

    if separate_actor_critic:
        actor_model.save(os.path.join(experiment_dir, f'actor_{episode_num}.h5'))
        target_actor.save(os.path.join(experiment_dir, f'target_actor_{episode_num}.h5'))
        critic_model.save(os.path.join(experiment_dir, f'critic_{episode_num}.h5'))
        target_critic.save(os.path.join(experiment_dir, f'target_critic_{episode_num}.h5'))
    else:
        model.save(os.path.join(experiment_dir, f'model_{episode_num}.h5'))
        target_model.save(os.path.join(experiment_dir, f'target_model_{episode_num}.h5'))
if __name__ == '__main__':

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = "./logs", profile_batch = '500,510')

    now = datetime.now()
    time_string = now.strftime("%d.%m.%Y %H:%M")

    offense_model = True
    separate_actor_critic = True
    dueling = False
    create_new_episodes = False
    train_just_critic = False

    sequence_length = Codes.sequence_length
    state_dim = 32 + 1 # + 1 for the is_attacking_the_left feature

    attacking_players_stats = 50
    defending_players_stats = 35

    # discrete_action_dim = 6 + 3 # 3 because 4 shots types -1 shot index
    discrete_action_dim = 6
    continuous_action_dim = 13

    if not offense_model:
        state_dim = state_dim + discrete_action_dim
        continuous_action_dim = 10

    # Learning rate for actor-critic models
    critic_lr = 0.0001
    actor_lr = 0.0002
    gamma = 0.99 # Discount factor for future rewards
    tau = 0.005 # Used to update target networks
    update_frequency = 1
    num_of_dense = 1
    batch_size = 16

    critic_optimizer = tf.keras.optimizers.SGD(critic_lr)
    actor_optimizer = tf.keras.optimizers.SGD(actor_lr)

    models_folder = r'models'

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    data_path = r'data\processed_data\merged_games'
    episodes_path = r'data\processed_data\episodes'

    # Create Experiment directory
    experiment_dir = os.path.join(models_folder, time_string.replace(':', '.'))
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    model_parameters_dict = {
        'sequence_length': sequence_length,
        'state dim': state_dim,
        'discrete actions dim': discrete_action_dim,
        'continuous actions dim': continuous_action_dim,
        'is offense model': offense_model,
        'critic learning rate': critic_lr,
        'actor learning rate': actor_lr,
        'gamma': gamma,
        'tau': tau,
        'batch size': batch_size,
        'update frequency': update_frequency,
        'number of dense layers': num_of_dense,
        'critic optimizer': 'SGD',
        'actor optimizer': 'SGD',
        'is using separate actor critic loss': separate_actor_critic,
        'just critic': train_just_critic,
        'dueling': dueling
    }

    with open(os.path.join(experiment_dir, 'parameters'), 'w') as file:
         file.write(json.dumps(model_parameters_dict, indent=2))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_dir, histogram_freq=1)

    if separate_actor_critic and not train_just_critic:
        actor = ActorNetwork(state_dim=state_dim, discrete_action_dim=discrete_action_dim,
                             continuous_action_dim=continuous_action_dim, sequence_length=sequence_length)

        critic = CriticNetwork(state_dim=state_dim, discrete_action_dim=discrete_action_dim,
                               continuous_action_dim=continuous_action_dim, sequence_length=sequence_length)

        actor_model = actor.create_actor_network()
        target_actor = actor.create_actor_network()

        critic_model = critic.create_critic_network()
        target_critic = critic.create_critic_network()

        # Making the weights equal initially
        target_actor.set_weights(actor_model.get_weights())
        target_critic.set_weights(critic_model.get_weights())

    ddpg_model = ActorCriticNetwork(state_dim=state_dim, discrete_action_dim=discrete_action_dim,
                         continuous_action_dim=continuous_action_dim, sequence_length=sequence_length)
    if offense_model:
        model = ddpg_model.create_model()
        target_model = ddpg_model.create_model()
    else:
        model = ddpg_model.create_defensive_agent_model()
        target_model = ddpg_model.create_defensive_agent_model()

    target_model.set_weights(model.get_weights())

    data_loader = Data_Loader(
        data_path,
        episodes_save_path=episodes_path,
        sequence_length=sequence_length,
        load_games_num=1000,
        transpose=False,
        skip_frames=6,
        create_new_episodes_data=create_new_episodes,
        offense=offense_model
    )

    buffer = Buffer(buffer_capacity=50000, batch_size=batch_size, states_dim=state_dim,
                    discrete_actions_dim=discrete_action_dim, continuous_actions_dim=continuous_action_dim,
                    actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer, sequence_length=sequence_length,
                    just_critic=train_just_critic, dueling=dueling)

    episode = data_loader.get_episode_data()
    episode_count = 1
    episode_check_point = 0
    start_storing_episode = episode_check_point - 5000

    # if episode_check_point != 0:
    #     load_pretrained_models()

    while episode:
        if len(episode) == 0:
            continue
        prev_state = episode[0]['state']
        if episode_count > start_storing_episode:
            for i in range(1, len(episode)):

                state, action, reward, done = episode[i].values()
                buffer.record(prev_state, action, reward, state)

                if episode_count > episode_check_point:
                    if separate_actor_critic:
                        buffer.learn(gamma, actor_model, target_actor, critic_model, target_critic, single_model=False)
                    else:
                        buffer.learn(gamma, model, target_model, single_model=True)

                    if i % update_frequency == 0:
                        if separate_actor_critic:
                            update_target(target_actor.variables, actor_model.variables, tau)
                            update_target(target_critic.variables, critic_model.variables, tau)
                        else:
                            update_target(target_model.variables, model.variables, tau)

                prev_state = state

            if episode_count % 100 == 0 and episode_count != 0:
                print(f'Learned {episode_count} episodes')

            if episode_count % 10000 == 0 and episode_count != 0:
                save_models(experiment_dir=experiment_dir, episode_num=str(episode_count))

        episode = data_loader.get_episode_data()
        episode_count += 1

    print(f'Skipped episodes number = {data_loader.skipped_episodes}')
    save_models(experiment_dir=experiment_dir, episode_num=str(episode_count) + '_done', done=True)
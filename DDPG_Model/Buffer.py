import tensorflow as tf
import numpy as np

class Buffer:
    def __init__(self, states_dim, discrete_actions_dim, continuous_actions_dim, actor_optimizer, critic_optimizer,
                 buffer_capacity=100000, batch_size=64, sequence_length=25, just_critic=False, dueling=False):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.just_critic = just_critic
        self.dueling = dueling

        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.attacking_players_buffer = np.zeros((self.buffer_capacity, 5))
        self.attacking_team_id_buffer = np.zeros((self.buffer_capacity, 1))
        self.attacking_players_stats_buffer = np.zeros((self.buffer_capacity, 50))

        self.defending_players_buffer = np.zeros((self.buffer_capacity, 5))
        self.defending_team_id_buffer = np.zeros((self.buffer_capacity, 1))
        self.defending_players_stats_buffer = np.zeros((self.buffer_capacity, 35))

        self.meta_data_buffer = np.zeros((self.buffer_capacity, 2))

        self.state_buffer = np.zeros((self.buffer_capacity, self.sequence_length, states_dim))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.sequence_length, states_dim))

        self.discrete_action_buffer = np.zeros((self.buffer_capacity, discrete_actions_dim))
        self.continuous_action_buffer = np.zeros((self.buffer_capacity, continuous_actions_dim))

        self.reward_buffer = np.zeros((self.buffer_capacity, 1))

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.huber = tf.keras.losses.Huber()

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, prev_state, action, reward, state):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        meta_data = state[0]
        state_seq = state[1]

        self.attacking_players_buffer[index] = meta_data['attacking_players']
        self.attacking_team_id_buffer[index] = meta_data['attacking_team_id']
        self.attacking_players_stats_buffer[index] = meta_data['attacking_players_stats']

        self.defending_players_buffer[index] = meta_data['defending_players']
        self.defending_team_id_buffer[index] = meta_data['defending_team_id']
        self.defending_players_stats_buffer[index] = meta_data['defending_players_stats']

        self.meta_data_buffer[index] = meta_data['meta_data']

        self.state_buffer[index] = prev_state[1]
        self.next_state_buffer[index] = state_seq

        self.discrete_action_buffer[index] = action[0]
        self.continuous_action_buffer[index] = action[1]

        self.reward_buffer[index] = reward

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def learn(self, gamma, actor_model, target_actor, critic_model=None, target_critic=None, single_model=False):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        attacking_players_batch = tf.convert_to_tensor(self.attacking_players_buffer[batch_indices], dtype=tf.float32)
        attacking_team_id_batch = tf.convert_to_tensor(self.attacking_team_id_buffer[batch_indices], dtype=tf.float32)

        attacking_players_stats_batch = \
            tf.convert_to_tensor(self.attacking_players_stats_buffer[batch_indices], dtype=tf.float32)

        defending_players_batch = tf.convert_to_tensor(self.defending_players_buffer[batch_indices], dtype=tf.float32)
        defending_team_id_batch = tf.convert_to_tensor(self.defending_team_id_buffer[batch_indices], dtype=tf.float32)

        defending_players_stats_batch = \
            tf.convert_to_tensor(self.defending_players_stats_buffer[batch_indices], dtype=tf.float32)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)

        discrete_action_batch = tf.convert_to_tensor(self.discrete_action_buffer[batch_indices], dtype=tf.float32)
        continuous_action_batch = tf.convert_to_tensor(self.continuous_action_buffer[batch_indices], dtype=tf.float32)

        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)

        if single_model:
            self.update_single_model(attacking_players_batch, attacking_team_id_batch, defending_players_batch,
                                     defending_team_id_batch, state_batch, next_state_batch, discrete_action_batch,
                                     continuous_action_batch, reward_batch, actor_model, target_actor, gamma)
        else:
            self.update(attacking_players_batch, attacking_team_id_batch, attacking_players_stats_batch,
                        defending_players_batch, defending_team_id_batch, defending_players_stats_batch,
                        state_batch, next_state_batch, discrete_action_batch, continuous_action_batch, reward_batch,
                        actor_model, target_actor, critic_model, target_critic, gamma)

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, attacking_players_batch, attacking_team_id_batch, attacking_players_stats_batch,
              defending_players_batch, defending_team_id_batch, defending_players_stats_batch,
                    state_batch, next_state_batch, discrete_action_batch, continuous_action_batch, reward_batch,
                    actor_model, target_actor, critic_model, target_critic, gamma
    ):

        # critic inputs:
        attacking_players_batches = list()
        defending_players_batches = list()

        for i in range(5):
            attacking_players_batches.append(attacking_players_batch[:, i:i+1])
            defending_players_batches.append(defending_players_batch[:, i:i+1])


        with tf.GradientTape() as tape:
            target_critic_inputs = [next_state_batch, discrete_action_batch, continuous_action_batch,
                                    attacking_team_id_batch, defending_team_id_batch, attacking_players_stats_batch,
                                    defending_players_stats_batch] + attacking_players_batches + defending_players_batches

            critic_value = target_critic(target_critic_inputs, training=True)
            y = reward_batch + tf.cast(gamma, tf.float32) * critic_value

            critic_inputs = [next_state_batch, discrete_action_batch, continuous_action_batch, attacking_team_id_batch,
                             defending_team_id_batch, attacking_players_stats_batch, defending_players_stats_batch] \
                                   + attacking_players_batches + defending_players_batches

            critic_value = critic_model(critic_inputs, training=True)
            critic_loss = self.huber(y, critic_value)

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:

            discrete_actions, continuous_actions = actor_model(attacking_players_batches + defending_players_batches +
                                  [state_batch, attacking_team_id_batch, defending_team_id_batch, attacking_players_stats_batch,
                                    defending_players_stats_batch], training=True)

            critic_inputs = [state_batch, discrete_actions, continuous_actions, attacking_team_id_batch,
                             defending_team_id_batch, attacking_players_stats_batch, defending_players_stats_batch]\
                            + attacking_players_batches + defending_players_batches

            critic_value = critic_model(critic_inputs, training=True)

            # Used `-value` as we want to maximize the value given
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update_single_model(
        self, attacking_players_batch, attacking_team_id_batch, defending_players_batch, defending_team_id_batch,
                    state_batch, next_state_batch, discrete_action_batch, continuous_action_batch, reward_batch,
                    model, target_model, gamma
    ):

        # Training and updating Actor & Critic networks.

        # critic inputs:
        attacking_players_batches = list()
        defending_players_batches = list()

        for i in range(5):
            attacking_players_batches.append(attacking_players_batch[:, i:i+1])
            defending_players_batches.append(defending_players_batch[:, i:i+1])


        with tf.GradientTape() as tape:
            model_inputs = attacking_players_batches + defending_players_batches + \
                           [next_state_batch, attacking_team_id_batch, defending_team_id_batch]

            if self.just_critic:
                model_inputs = model_inputs + [discrete_action_batch, continuous_action_batch]
                target_critic_value = target_model(model_inputs, training=True)
            else:
                _, _, target_critic_value = target_model(model_inputs, training=True)

            y = reward_batch + tf.cast(gamma, tf.float32) * target_critic_value

            if self.just_critic:
                critic_value = model(model_inputs, training=True)
            else:
                _, _, critic_value = model(model_inputs, training=True)

            # Change to MSE
            critic_loss = self.huber(y, critic_value)

        critic_grad = tape.gradient(critic_loss, model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, model.trainable_variables)
        )

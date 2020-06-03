import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.tf_layers import linear


class PvgPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTM cells for processing complex observations

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param hidden_size: (int) The number of units in Lstm cell (if None, default to 100)
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """
    
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, hidden_size=100,
                 act_fun=tf.tanh, **kwargs):
        super(PvgPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=False)

        with tf.variable_scope("model", reuse=reuse):
            n_acts = ac_space.n

            # Placeholder for a variable number of observations with undefined length
            # Each character is presented in 50 nodes
            self._obs_ph = tf.placeholder(shape=[None, None,50], dtype=tf.float32, name='obs_ph')
            # two lstm cells are created which are merged for a more powerful cell
            lstm_cell_1 = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden_size)
            lstm_cell_2 = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden_size)
            multi_lstm_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell_1,lstm_cell_2], state_is_tuple=True)
            # Create the recurent neural network
            _, final_state = tf.compat.v1.nn.dynamic_rnn(cell=multi_lstm_cells, inputs=self.obs_ph, dtype=tf.float32)
            # Create policy and value layers
            pi_latent = linear(final_state[-1][-1],"pi_logits", n_acts)
            vf_latent = linear(final_state[-1][-1],"vf_logits", 1)
            self._value_fn = linear(vf_latent, 'vf', 1)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                               {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp
    
    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

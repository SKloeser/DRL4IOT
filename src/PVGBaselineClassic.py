import tensorflow as tf
import numpy as np
import math
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class PVGBaselineClassic():
    
    def __init__(self):
        pass

    def linear(self,input_, output_size, name, init_bias=0.0):
        shape = input_.get_shape().as_list()
        with tf.variable_scope(name):
            W = tf.get_variable("weights", [shape[-1], output_size], tf.float32, tf.random_normal_initializer(stddev=1.0 / math.sqrt(shape[-1])))
        if init_bias is None:
            return tf.matmul(input_, W)
        with tf.variable_scope(name):
            b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(input_, W) + b

    def mlp(self,x, sizes, activation=tf.tanh, output_activation=None):
        dummy_size = sizes[:-1]
        for size in dummy_size:
            x = tf.layers.dense(x, units=size, activation=activation)
        return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

    def reward_to_go(self,rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    def train(self,env, hidden_sizes=[100], lr=1e-2,epochs=100, batch_size=5000, render=False):
        obs_dim = env.observation_space.shape[0]
        n_acts = env.action_space.n

        obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
        logits = self.mlp(obs_ph, sizes=hidden_sizes+[n_acts])
        actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
        weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
        act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
        action_masks = tf.one_hot(act_ph, n_acts)
        log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)

        loss = -tf.reduce_mean(weights_ph * log_probs)
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        def train_one_epoch():
            batch_obs = []
            batch_acts = []
            batch_weights = []
            batch_rets = []
            batch_lens = []
            obs = env.reset()
            ep_rews = []
            finished_rendering_this_epoch = False
            while True:
                if (not finished_rendering_this_epoch) and render:
                    env.render()
                obs_old = obs.copy()
                act = sess.run(actions, {obs_ph: [obs]})[0]
                obs, rew, done, _ = env.step(act)
                batch_obs.append(obs_old)
                batch_acts.append(act)
                ep_rews.append(rew)

                if done:
                    ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                    batch_rets.append(ep_ret)
                    batch_lens.append(ep_len)
                    batch_weights += list(self.reward_to_go(ep_rews))
                    obs, done, ep_rews = env.reset(), False, []
                    finished_rendering_this_epoch = True
                    if len(batch_obs) > batch_size:
                        break

            baseline = np.median(np.array(batch_weights))
            advantage = np.array(batch_weights) - baseline
            advantage = advantage.tolist()

            batch_loss, _  = sess.run([loss, train_op],
                                    feed_dict={
                                        obs_ph: np.array(batch_obs),
                                        act_ph: np.array(batch_acts),
                                        weights_ph: np.array(advantage)
                                    })

            return batch_loss, batch_rets, batch_lens

        average_returns = []
        std_returns = []

        for i in range(epochs):
            batch_loss, batch_rets, batch_lens = train_one_epoch()
            print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                    (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
            average_returns.append(np.mean(batch_rets))
            std_returns.append(np.std(batch_rets))

        sess.close()
        tf.reset_default_graph()
        
        return average_returns, std_returns
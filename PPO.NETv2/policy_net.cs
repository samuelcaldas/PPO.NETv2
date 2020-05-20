// import gym
// import numpy as np
// import tensorflow.compat.v1 as tf
// tf.disable_v2_behavior()

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Tensorflow.Binding;
using NumSharp;
using SixLabors.ImageSharp;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Rendering.WinForm;
using Tensorflow;
using Gym.Spaces;

namespace PPO.NETv2
{
    public class Policy_net
    {
        public Tensor obs;
        public Tensor act_probs;
        public Tensor v_preds;
        private Tensor act_stochastic;
        private Tensor act_deterministic;
        private string scope;

        public Policy_net(string name, CartPoleEnv env, double temp = 0.1)
        {
            /*
            :param name: string
            :param env: gym env
            :param temp: temperature of boltzmann distribution
            */

            var ob_space = env.ObservationSpace;
            var act_space = env.ActionSpace;

            using (tf.variable_scope(name))
            {
                this.obs = tf.placeholder(dtype: tf.float32, shape: new TensorShape((Unknown),(4)), name: "obs");

                using (tf.variable_scope("policy_net"))
                {
                    var layer_1 = tf.layers.dense(inputs: this.obs, units: 20, activation: tf.nn.tanh());
                    var layer_2 = tf.layers.dense(inputs: layer_1, units: 20, activation: tf.nn.tanh());
                    var layer_3 = tf.layers.dense(inputs: layer_2, units: 4, activation: tf.nn.tanh());
                    this.act_probs = tf.layers.dense(inputs: tf.divide(layer_3, new Tensor(temp)), units: 4, activation: tf.nn.softmax());
                }
                using (tf.variable_scope("value_net"))
                {
                    var layer_1 = tf.layers.dense(inputs:   this.obs,   units: 20,  activation: tf.nn.tanh());
                    var layer_2 = tf.layers.dense(inputs:   layer_1,    units: 20,  activation: tf.nn.tanh());
                    this.v_preds = tf.layers.dense(inputs:  layer_2,    units: 1,   activation: null);
                }
                this.act_stochastic = tf.random.categorical(tf.log(this.act_probs), num_samples: 1);
                this.act_stochastic = tf.reshape(this.act_stochastic, shape:(-1));

                this.act_deterministic = tf.argmax(this.act_probs, axis: 1);

                this.scope = tf.get_variable_scope().name;
            }
        }
        public (string, string) act(NDArray obs, bool stochastic = true)
        {
            if (stochastic)
            {
                NDArray result = tf.get_default_session().run(new[] { this.act_stochastic, this.v_preds }, feed_dict: new FeedItem[] { new FeedItem(this.obs, obs) });
                var act = result.GetItem(0);
                var v_pred = result.GetItem(1);
                return (act, v_pred);
            }
            else
            {
                NDArray result = tf.get_default_session().run(new[] { this.act_deterministic, this.v_preds }, feed_dict: new FeedItem[] { new FeedItem(this.obs, obs) });
                string act      = result.GetItem(0);
                string v_pred   = result.GetItem(1);
                return (act, v_pred);
            }
        }
        public NDArray get_action_prob(NDArray obs)
        {
            return tf.get_default_session().run(this.act_probs, feed_dict: new FeedItem[] { new FeedItem(this.obs, obs) });
        }
        public List<RefVariable> get_variables()
        {
            return tf.get_collection<RefVariable>(tf.GraphKeys.GLOBAL_VARIABLES, this.scope);
        }
        public List<RefVariable> get_trainable_variables()
        {
            return tf.get_collection<RefVariable>(tf.GraphKeys.TRAINABLE_VARIABLES, this.scope);
        }

    }
}

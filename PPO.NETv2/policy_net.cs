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

namespace PPO.NETv2
{
    class Policy_net{
    public void Policy_net(name){ str, env, temp=0.1)){
        /*
        :param name: string
        :param env: gym env
        :param temp: temperature of boltzmann distribution
        */

        ob_space = env.observation_space
        act_space = env.action_space

        using (tf.variable_scope(name)){
            this.obs = tf.placeholder(dtype=tf.dlouble32, shape=[None] + list(ob_space.shape), name="obs")

            using (tf.variable_scope("policy_net")){
                layer_1         = tf.layers.dense(inputs=this.obs, units=20, activation=tf.tanh)
                layer_2         = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                layer_3         = tf.layers.dense(inputs=layer_2, units=act_space.n, activation=tf.tanh)
                this.act_probs  = tf.layers.dense(inputs=tf.divide(layer_3, temp), units=act_space.n, activation=tf.nn.softmax)

            using (tf.variable_scope("value_net")){
                layer_1         = tf.layers.dense(inputs=this.obs, units=20, activation=tf.tanh)
                layer_2         = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                this.v_preds    = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            this.act_stochastic = tf.random.categorical(tf.log(this.act_probs), num_samples=1)
            this.act_stochastic = tf.reshape(this.act_stochastic, shape=[-1])

            this.act_deterministic = tf.argmax(this.act_probs, axis=1)

            this.scope = tf.get_variable_scope().name

    public void act(obs, stochastic=true)){
        if (stochastic){
            return tf.get_default_session().run([this.act_stochastic, this.v_preds], feed_dict={this.obs){ obs})
        }else{
            return tf.get_default_session().run([this.act_deterministic, this.v_preds], feed_dict={this.obs){ obs})

    public void get_action_prob(obs)){
        return tf.get_default_session().run(this.act_probs, feed_dict={this.obs){ obs})

    public void get_variables(this)){
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, this.scope)

    public void get_trainable_variables(this)){
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, this.scope)


// import tensorflow.compat.v1 as tf
// tf.disable_v2_behavior()
// import copy

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

    class PPOTrain{
    public void PPOTrain(Policy, Old_Policy, gamma=0.95, clip_value=0.2, c_1=1, c_2=0.01){
        /*
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for (value difference
        :param c_2: parameter for (entropy bonus
        */

        this.Policy     = Policy
        this.Old_Policy = Old_Policy
        this.gamma      = gamma

        pi_trainable = this.Policy.get_trainable_variables()
        old_pi_trainable = this.Old_Policy.get_trainable_variables()

        // assign_operations for (policy parameter values to old policy parameters
        // atribuir operações para valores de parâmetros de política a parâmetros de política antigos
        using (tf.variable_scope("assign_op")){
            this.assign_ops = new object[]{};
            for (v_old, v in zip(old_pi_trainable, pi_trainable)){
                this.assign_ops.append(tf.assign(v_old, v))

        // inputs for (train_op
        // inputs para train_op
        using (tf.variable_scope("train_inp")){
            this.actions        = tf.placeholder(dtype=tf.int32, shape=[None], name="actions")
            this.rewards        = tf.placeholder(dtype=tf.dlouble32, shape=[None], name="rewards")
            this.v_preds_next   = tf.placeholder(dtype=tf.dlouble32, shape=[None], name="v_preds_next")
            this.gaes           = tf.placeholder(dtype=tf.dlouble32, shape=[None], name="gaes")

        act_probs = this.Policy.act_probs
        act_probs_old = this.Old_Policy.act_probs

        // probabilities of actions which agent took using (policy
        // probabilidades de ações que o agente executou com a política
        act_probs = act_probs * tf.one_hot(indices=this.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        // probabilities of actions which agent took using (old policy
        // probabilidades de ações que o agente executou com a política antiga
        act_probs_old = act_probs_old * tf.one_hot(indices=this.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        using (tf.variable_scope("loss/clip")){
            // ratios = tf.divide(act_probs, act_probs_old)
            ratios          = tf.exp(tf.log(act_probs);- tf.log(act_probs_old))
            clipped_ratios  = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            loss_clip       = tf.minimum(tf.multiply(this.gaes, ratios), tf.multiply(this.gaes, clipped_ratios))
            loss_clip       = tf.reduce_mean(loss_clip)
            tf.summary.scalar("loss_clip", loss_clip)

        // construct computation graph for (loss of value function
        // constrói gráfico de cálculo para perda da função de valor
        using (tf.variable_scope("loss/vf")){
            v_preds = this.Policy.v_preds
            loss_vf = tf.squared_difference(this.rewards + this.gamma * this.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)
            tf.summary.scalar("loss_vf", loss_vf)

        // construct computation graph for (loss of entropy bonus
        // construir gráfico de computação para perda de bônus de entropia
        using (tf.variable_scope("loss/entropy")){
            entropy = -tf.reduce_sum(this.Policy.act_probs * tf.log(tf.clip_by_value(this.Policy.act_probs, 1e-10, 1.0)), axis=1)
            entropy = tf.reduce_mean(entropy, axis=0);  // mean of entropy of pi(obs)
                                                        // média de entropia de pi (obs)
            tf.summary.scalar("entropy", entropy)

        using (tf.variable_scope("loss")){
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy
            loss = -loss  // minimize -loss == maximize loss
            tf.summary.scalar("loss", loss)

        this.merged     = tf.summary.merge_all()
        optimizer       = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        this.train_op   = optimizer.minimize(loss, var_list=pi_trainable)

    public void train(obs, actions, rewards, v_preds_next, gaes)){ // Função de treinamento
        tf.get_default_session().run(
            [this.train_op], 
            feed_dict={
                this.Policy.obs){ obs,
                this.Old_Policy.obs){ obs,
                this.actions){ actions,
                this.rewards){ rewards,
                this.v_preds_next){ v_preds_next,
                this.gaes){ gaes
            }
        )

    public void get_summary(obs, actions, rewards, v_preds_next, gaes)){   // Obtém o sumário
        return tf.get_default_session().run(
            [this.merged], 
            feed_dict={
                this.Policy.obs){ obs,
                this.Old_Policy.obs){ obs,
                this.actions){ actions,
                this.rewards){ rewards,
                this.v_preds_next){ v_preds_next,
                this.gaes){ gaes
            }
        )

    public void assign_policy_parameters(this)){
        // assign policy parameter values to old policy parameters
        // Atribuir valores de parâmetro de política a parâmetros de política antigos
        return tf.get_default_session().run(this.assign_ops)

    public void get_gaes(rewards, v_preds, v_preds_next)){
        deltas = new object[]{};
        for (r_t, v_next, v in zip(rewards, v_preds_next, v_preds)){
            deltas.append(r_t + this.gamma * v_next - v)
        // calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        // calcular o estimador de vantagem generativa (lambda = 1), consulte o documento ppo eq(11)
        //gaes = copy.deepcopy(deltas)
        gaes = deltas
        for (t in reversed(range(len(gaes);- 1))){    // é T-1, onde T é o intervalo de tempo que executa a política
            gaes[t] = gaes[t] + this.gamma * gaes[t + 1]
        return gaes


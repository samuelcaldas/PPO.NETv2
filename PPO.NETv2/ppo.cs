// import tensorflow.compat.v1 as tf
// tf.disable_v2_behavior()
// import copy

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;
using SixLabors.ImageSharp;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Rendering.WinForm;
using static Tensorflow.Binding;
using Tensorflow;

namespace PPO.NETv2
{
    class PPOTrain
    {
        double gamma;
        Policy_net Policy;
        Policy_net Old_Policy;
        private List<Tensor> assign_ops;
        private Tensor actions;
        private Tensor rewards;
        private Tensor v_preds_next;
        private Tensor gaes;
        private Tensor loss_clip;
        private Tensor loss_vf;
        private Tensor entropy;
        private Tensor loss;
        private Operation train_op;
        private Tensor merged;

        public PPOTrain(Policy_net policy, Policy_net old_policy, double gamma = 0.95, double clip_value = 0.2, double c_1 = 1, double c_2 = 0.01)
        {
            /*
            :param Policy:
            :param Old_Policy:
            :param gamma:
            :param clip_value:
            :param c_1: parameter for value difference
            :param c_2: parameter for entropy bonus
            */
            Policy = policy;
            Old_Policy = old_policy;
            this.gamma= gamma;
            List<RefVariable> pi_trainable = Policy.get_trainable_variables();
            List<RefVariable> old_pi_trainable = Old_Policy.get_trainable_variables();

            // assign_operations for (policy parameter values to old policy parameters
            // atribuir operações para valores de parâmetros de política a parâmetros de política antigos
            using (tf.variable_scope("assign_op"))
            {
                foreach ((var v_old, var v) in zip(old_pi_trainable, pi_trainable))
                {
                    this.assign_ops.add(tf.assign(v_old, v));
                }
            }
            // inputs for (train_op
            // inputs para train_op
            using (tf.variable_scope("train_inp"))
            {
                this.actions = tf.placeholder(dtype: tf.int32, Shape:[null], name: "actions");
                this.rewards = tf.placeholder(dtype: tf.float32, Shape:[null], name: "rewards");
                this.v_preds_next = tf.placeholder(dtype: tf.float32, Shape:[null], name: "v_preds_next");
                this.gaes = tf.placeholder(dtype: tf.float32, Shape:[null], name: "gaes");
            }
            Tensor act_probs = Policy.act_probs;
            Tensor act_probs_old = Old_Policy.act_probs;

            // probabilities of actions which agent took using (policy
            // probabilidades de ações que o agente executou com a política
            act_probs = act_probs * tf.one_hot(indices: this.actions, depth: act_probs.Shape[1]);
            act_probs = tf.reduce_sum(act_probs, axis: 1);

            // probabilities of actions which agent took using (old policy
            // probabilidades de ações que o agente executou com a política antiga
            act_probs_old = act_probs_old * tf.one_hot(indices: this.actions, depth: act_probs_old.Shape[1]);
            act_probs_old = tf.reduce_sum(act_probs_old, axis: 1);

            using (tf.variable_scope("loss/clip"))
            {
                // ratios = tf.divide(act_probs, act_probs_old)
                Tensor ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old));
                Tensor clipped_ratios = tf.clip_by_value(ratios, clip_value_min: new Tensor(1 - clip_value), clip_value_max: new Tensor(1 + clip_value));
                loss_clip = tf.minimum(tf.multiply(this.gaes, ratios), tf.multiply(this.gaes, clipped_ratios));
                loss_clip = tf.reduce_mean(loss_clip);
                tf.summary.scalar("loss_clip", loss_clip);
            }
            // construct computation graph for (loss of value function
            // constrói gráfico de cálculo para perda da função de valor
            using (tf.variable_scope("loss/vf"))
            {
                Tensor v_preds = Policy.v_preds;
                loss_vf = tf.squared_difference(this.rewards + gamma * this.v_preds_next, v_preds);
                loss_vf = tf.reduce_mean(loss_vf);
                tf.summary.scalar("loss_vf", loss_vf);
            }
            // construct computation graph for (loss of entropy bonus
            // construir gráfico de computação para perda de bônus de entropia
            using (tf.variable_scope("loss/entropy"))
            {
                entropy = -tf.reduce_sum(Policy.act_probs * tf.log(tf.clip_by_value(Policy.act_probs, new Tensor(1e-10), new Tensor(1.0))), axis: 1);
                entropy = tf.reduce_mean(entropy, axis: 0);  // mean of entropy of pi(obs)
                                                             // média de entropia de pi (obs)
                tf.summary.scalar("entropy", entropy);
            }
            using (tf.variable_scope("loss"))
            {
                loss = loss_clip - c_1 * loss_vf + c_2 * entropy;
                loss = -loss;  // minimize -loss == maximize loss
                tf.summary.scalar("loss", loss);
            }
            this.merged = tf.summary.merge_all();
            Optimizer optimizer = tf.train.AdamOptimizer(learning_rate: 1e-4, epsilon: 1e-5);
            this.train_op = optimizer.minimize(loss, var_list: pi_trainable);
        }
        public void train(string obs, string actions, string rewards, string v_preds_next, string gaes)
        {
            tf.get_default_session().run(
                new[] { this.train_op },
                feed_dict: new FeedItem[] {
                    new FeedItem(Policy.obs, obs),
                    new FeedItem(Old_Policy.obs, obs),
                    new FeedItem(this.actions, actions),
                    new FeedItem(this.rewards, rewards),
                    new FeedItem(this.v_preds_next, v_preds_next),
                    new FeedItem(this.gaes, gaes)
                }
            );
        }
        public NDArray get_summary(string obs, string actions, string rewards, string v_preds_next, string gaes)
        {
            return tf.get_default_session().run(
                new[] { this.merged },
                feed_dict: new FeedItem[] {
                    new FeedItem(this.Policy.obs, obs),
                    new FeedItem(this.Old_Policy.obs, obs),
                    new FeedItem(this.actions, actions),
                    new FeedItem(this.rewards, rewards),
                    new FeedItem(this.v_preds_next, v_preds_next),
                    new FeedItem(this.gaes, gaes)
                }
            );
        }
        public NDArray assign_policy_parameters()
        {
            // assign policy parameter values to old policy parameters
            // Atribuir valores de parâmetro de política a parâmetros de política antigos
            return tf.get_default_session().run(this.assign_ops);
        }
        public void get_gaes(List<double> rewards, Tensor v_preds, Tensor v_preds_next)
        {
            List<string> gaes = new List<string>();
            foreach ((string r_t, string v_next, string v) in zip<double, string, string>(rewards, v_preds_next, v_preds))
            {
                gaes.add(r_t + this.gamma * v_next - v);
            }
            // calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
            // calcular o estimador de vantagem generativa (lambda = 1), consulte o documento ppo eq(11)
            for (var t = gaes.Count(); t > 0; t--)
            {    // é T-1, onde T é o intervalo de tempo que executa a política
                gaes[t] = gaes[t] + this.gamma * gaes[t + 1];
            }
            return gaes;
        }
    }
}



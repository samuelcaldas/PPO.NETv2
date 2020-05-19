//// test
// import gym
// import numpy as np
// import tensorflow.compat.v1 as tf
// tf.disable_v2_behavior()
// from policy_net import Policy_net
// from ppo import PPOTrain

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
    class Test_policy
    {
        ITERATION = int(3 * 10e5)
GAMMA = 0.95


        static void _Main(string[] args);
        {
            CartPoleEnv env = new CartPoleEnv(WinFormEnvViewer.Factory); //or AvaloniaEnvViewer.Factory
        env.seed(0)
    ob_space = env.observation_space
    Policy = Policy_net("policy", env)
    Old_Policy = Policy_net("old_policy", env)
    PPO = PPOTrain(Policy, Old_Policy, gamma=GAMMA)
    saver = tf.train.Saver()

    using(var sess = tf.Session()){
        writer = tf.summary.FileWriter("./log/test", sess.graph)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "model/model.ckpt")
        obs = env.reset()
        reward = 0
        success_num = 0

        for (iteration in range(ITERATION)){  // episode
            observations = new object[]{};
            actions = new object[]{};
            v_preds = new object[]{};
            rewards = new object[]{};
            run_policy_steps = 0
            env.render()
            while (true){  // run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.dlouble32); // prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=false)

                act     = act.item()
                v_pred  = v_pred.item()

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                next_obs, reward, done, info = env.step(act)

                if (done){
                    v_preds_next = v_preds[1){] + [0]  // next state of terminate state has 0 state value
                    obs = env.reset()
                    reward = -1
                    break;
                }else{
                    obs = next_obs
                }
            //writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="episode_length", simple_value=run_policy_steps)]), iteration);
            //writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=sum(rewards))]), iteration);

            // end condition of test
            if (sum(rewards)>= 195){
                success_num += 1
                if (success_num >= 100){
                    Console.WriteLine("Iteration){ ", iteration)
                    Console.WriteLine("Clear!!")
                    break;
                }
            }else{
                success_num = 0
            }

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            // convert list to numpy array for (feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)
            rewards = np.array(rewards).astype(dtype=np.dlouble32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.dlouble32)
            gaes = np.array(gaes).astype(dtype=np.dlouble32)
            gaes = (gaes - gaes.mean());/ gaes.std()

            inp = [observations, actions, rewards, v_preds_next, gaes]

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      rewards=inp[2],
                                      v_preds_next=inp[3],
                                      gaes=inp[4])[0]

            writer.add_summary(summary, iteration)
        writer.close()

}
    }
}

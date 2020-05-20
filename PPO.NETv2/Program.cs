//// Train
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
using Tensorflow;
using Gym.Spaces;
using Tensorflow.Summaries;

namespace PPO.NETv2
{
    class Program
    {
        static void Main(string[] args)
        {
            List<NDArray> observations = new List<NDArray>();
            List<NDArray> actions = new List<NDArray>();
            List<NDArray> v_preds = new List<NDArray>();
            List<double> rewards = new List<double>();
            string v_preds_next;

            double EPISODES = 1e5;
            double GAMMA = 0.95;

            CartPoleEnv env = new CartPoleEnv(WinFormEnvViewer.Factory);//or AvaloniaEnvViewer.Factory  // Instancia o ambiente CartPole
            env.Seed(0);                                                //
            Space       ob_space = env.ObservationSpace;                // Descrevem o formato de observações válidas do espaço
            Policy_net  Policy      = new Policy_net("policy", env);    // Cria a rede de Politica
            Policy_net  Old_Policy  = new Policy_net("old_policy", env);// Cria a rede de politica antiga
            PPOTrain    PPO         = new PPOTrain(Policy, Old_Policy, gamma: GAMMA);
            Saver       saver       = tf.train.Saver();                 //

            using (var sess = tf.Session())  // Bloco da sessão
            {
                FileWriter writer = tf.summary.FileWriter("./log/train", sess.graph);   // Define diretório de logs
                sess.run(tf.global_variables_initializer());                            // Inicializa as redes

                NDArray obs = env.Reset();  // Reseta o ambiente e obtêm a primeira observação
                double reward = 0;          // Armazena as recompensas
                int success_num = 0;        // Contador de sucessos

                for (int episode = 0; episode < EPISODES; episode++))   // Loop do episodio
                {
                    int run_policy_steps = 0;   // Contador de passos em cada episodio
                    env.Render();               // Renderiza o ambiente

                    while (true) // Execute a política RUN_POLICY_STEPS, que é muito menor que a duração do episódio
                    {
                        run_policy_steps += 1;                               // Incrementa contador de passos de cada episodio
                        obs = np.stack(new[] { obs }).astype(dtype: np.float32);     // prepare to feed placeholder Policy.obs
                        var (act, v_pred) = Policy.act(obs: obs, stochastic: true); // Corre a rede neural e obtêm uma ação e o V previsto

                        act = act.item();           // Transforma um array do numpy 
                        v_pred = v_pred.item();    // em um objeto scalar do Python

                        observations.add(obs);   // Adiciona a observação ao buffer de observações
                        actions.add(act);        // Adiciona a ação ao buffer de ações
                        v_preds.add(v_pred);     // Adiciona a v_pred ao buffer de v_pred
                        rewards.add(reward);     // Adiciona a recompensa ao buffer de recompensa

                        (NDArray next_obs, double _reward, bool done, Gym.Collections.Dict info) = env.Step(act);   // envia a ação ao ambiente e recebe a próxima observação, a recompensa e se o passo terminou
                        reward = _reward;

                        if (done)
                        {                // Se o done for (verdadeiro ...

                            v_preds_next = v_preds[1:] + [0];    // [1:] seleciona do segundo elemento da lista em diante e + [0] adiciona um elemento de valor zero no final da lista
                                                                 // next state of terminate state has 0 state value
                                                                 // próximo estado do estado final tem 0 valor de estado
                            obs = env.Reset();  //   Redefine o ambiente
                            reward = -1;         //   define a recompensa como -1 (?)
                            break;               //   Sai do loop while
                        }
                        else
                        {                   // Senão...
                            obs = next_obs;      //   Armazena em obs a próxima observação
                        }
                    }
                    // Armazena em log para visualização no tensorboard
                    //writer.add_summary(tf.Summary(value:[tf.Summary.Value(tag:"episode_length", simple_value:run_policy_steps)]), episode);
                    //writer.add_summary(tf.Summary(value:[tf.Summary.Value(tag:"episode_reward", simple_value:rewards.Sum())]),     episode);

                    // Condicional para finalizar o teste
                    if (rewards.Sum() >= 195)
                    {                         // Se a soma das recompensas for (maior ou igual 195
                        success_num += 1;                            //   Incrementa o contador de sucessos
                        if (success_num >= 100)
                        {                      //   Se ocorrerem 100 sucessos
                            saver.save(sess, "./model/model.ckpt"); //       Salva a sessão
                            Console.WriteLine("Clear!! Model saved.");          //       Escreve na tela
                            break;                                   //       Sai do loop
                        }
                    }
                    else
                    {                                           // senão, 
                        success_num = 0;                             //   zera o contador de sucessos
                    }
                    Console.WriteLine("EP: ", episode, " Rw: ", rewards.Sum());     // Escreve na tela o numero do episodio e a recompensa

                    gaes = PPO.get_gaes(rewards: rewards, v_preds: v_preds, v_preds_next: v_preds_next);// ?

                // Converte lista em NPArray para alimentar o tf.placeholder
                    newShape=[-1] + list(ob_space.Shape);// cria um array [-1, 4]
                    observations = np.reshape(observations, newShape: newShape);// antes, cada linha de observations era um array idependente. depois do reshape, observations passou ser um array só com varias linhas.

                    actions = np.array(actions).astype(dtype: np.int32);

                    rewards = np.array(rewards).astype(dtype: np.float32);
                    v_preds_next = np.array(v_preds_next).astype(dtype: np.float32);
                    gaes = np.array(gaes).astype(dtype: np.float32);
                    gaes = (gaes - gaes.mean()) / gaes.std();// subtrai dos itens de gaes a media de todos os itens de gaes e divide todos pelo desvio padrao de gaes

                    PPO.assign_policy_parameters();

                    inp = new[] { observations, actions, rewards, v_preds_next, gaes };  // Cria um array com 5 colunas: observações, ações, recompensas, 

                    // Treina
                    for (epoch = 0; epoch < 4; epoch++)
                    {
                        sample_indices = np.random.randint(low: 0, high: observations.Shape[0], size: 64);// índices estão em [baixo, alto]
                        foreach (a in inp)
                        {
                            sampled_inp.add(np.take(a: a, indices: sample_indices, axis: 0));   // amostra de dados de treinamento
                        }
                        PPO.train(obs: sampled_inp[0], actions: sampled_inp[1], rewards: sampled_inp[2], v_preds_next: sampled_inp[3], gaes: sampled_inp[4]);
                    }
                    summary = PPO.get_summary(obs: inp[0], actions: inp[1], rewards: inp[2], v_preds_next: inp[3], gaes: inp[4])[0];

                    writer.add_summary(summary, episode);
                }
                writer.close(); // Final do episódio
            }
        }
    }
}
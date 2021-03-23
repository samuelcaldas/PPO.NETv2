﻿using Tensorflow.Keras.ArgsDefinition;

namespace Tensorflow.Keras.Optimizers
{
    public class OptimizerApi
    {
        /// <summary>
        /// Adam optimization is a stochastic gradient descent method that is based on
        /// adaptive estimation of first-order and second-order moments.
        /// </summary>
        /// <param name="learning_rate"></param>
        /// <param name="beta_1"></param>
        /// <param name="beta_2"></param>
        /// <param name="epsilon"></param>
        /// <param name="amsgrad"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public OptimizerV2 Adam(float learning_rate = 0.001f,
                float beta_1 = 0.9f,
                float beta_2 = 0.999f,
                float epsilon = 1e-7f,
                bool amsgrad = false,
                string name = "Adam")
            => new Adam(learning_rate: learning_rate,
                beta_1: beta_1,
                beta_2: beta_2,
                epsilon: epsilon,
                amsgrad: amsgrad,
                name: name);

        /// <summary>
        /// Construct a new RMSprop optimizer.
        /// </summary>
        /// <param name="learning_rate"></param>
        /// <param name="rho"></param>
        /// <param name="momentum"></param>
        /// <param name="epsilon"></param>
        /// <param name="centered"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public OptimizerV2 RMSprop(float learning_rate = 0.001f,
                float rho = 0.9f,
                float momentum = 0.0f,
                float epsilon = 1e-7f,
                bool centered = false,
                string name = "RMSprop")
            => new RMSprop(new RMSpropArgs
            {
                LearningRate = learning_rate,
                RHO = rho,
                Momentum = momentum,
                Epsilon = epsilon,
                Centered = centered,
                Name = name
            });

        public SGD SGD(float learning_rate)
            => new SGD(learning_rate);
    }
}

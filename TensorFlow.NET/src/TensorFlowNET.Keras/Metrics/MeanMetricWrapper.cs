﻿using System;

namespace Tensorflow.Keras.Metrics
{
    public class MeanMetricWrapper : Mean
    {
        string name;
        Func<Tensor, Tensor, Tensor> _fn = null;

        public MeanMetricWrapper(Func<Tensor, Tensor, Tensor> fn, string name, TF_DataType dtype = TF_DataType.TF_FLOAT)
            : base(name: name, dtype: dtype)
        {
            _fn = fn;
        }

        public override Tensor update_state(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
        {
            y_true = math_ops.cast(y_true, _dtype);
            y_pred = math_ops.cast(y_pred, _dtype);

            var matches = _fn(y_true, y_pred);
            return update_state(matches, sample_weight: sample_weight);
        }
    }
}

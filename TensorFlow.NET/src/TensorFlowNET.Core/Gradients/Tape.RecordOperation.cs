﻿using System;
using System.Collections.Generic;
using Tensorflow.Util;
using static Tensorflow.tensorflow;
using static Tensorflow.Binding;
using System.Linq;

namespace Tensorflow.Gradients
{
    public partial class Tape
    {
        long next_op_id_ = 0;
        UnorderedMap<long, long> tensor_usage_;

        public void RecordOperation(string op_type,
            Tensor[] input_tensors,
            TapeTensor[] output_tensors,
            Func<BackwardFunction> backward_function_getter)
        {
            var input_ids = input_tensors.Select(x => x.Id).ToArray();
            var input_dtypes = input_tensors.Select(x => x.dtype).ToArray();

            if (!ShouldRecord(input_ids, input_dtypes))
            {
                return;
            }

            long op_id = next_op_id_++;
            var ids = new List<long>(input_ids.Length);
            foreach (var i in input_ids)
            {
                tensor_usage_[i]++;
                ids.Add(i);
            }

            var tensors = new List<TapeTensor>(output_tensors.Length);
            foreach (var o in output_tensors)
            {
                tensor_tape_[o.GetID()] = op_id;
                tf.Logger.Debug($"RecordOperation: tensor_tape_[{o.GetID()}] = {op_id}");
                tensor_usage_[o.GetID()] = 1;
                tensors.Add(o);
            }

            op_tape_[op_id] = new OpTapeEntry<BackwardFunction, TapeTensor>
            {
                op_type = op_type,
                output_tensor_info = tensors.ToArray(),
                input_tensor_id = ids.ToArray(),
                backward_function = backward_function_getter()
            };
        }
    }
}

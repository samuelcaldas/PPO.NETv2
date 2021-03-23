﻿namespace Tensorflow.Gradients
{
    /// <summary>
    /// Represents an entry in the tape.
    /// </summary>
    /// <typeparam name="BackwardFunction"></typeparam>
    /// <typeparam name="TapeTensor"></typeparam>
    public class OpTapeEntry<BackwardFunction, TapeTensor>
    {
        public string op_type { get; set; }
        public TapeTensor[] output_tensor_info { get; set; }
        public long[] input_tensor_id { get; set; }
        public BackwardFunction backward_function { get; set; }
        public override string ToString()
            => $"{op_type}, inputs: {string.Join(",", input_tensor_id)}";
    }
}

﻿using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Keras.Utils
{
    public class np_utils
    {
        /// <summary>
        /// Converts a class vector (integers) to binary class matrix.
        /// </summary>
        /// <param name="y"></param>
        /// <param name="num_classes"></param>
        /// <param name="dtype"></param>
        /// <returns></returns>
        public static NDArray to_categorical(NDArray y, int num_classes = -1, TF_DataType dtype = TF_DataType.TF_FLOAT)
        {
            var y1 = y.astype(NPTypeCode.Int32).ToArray<int>();
            // var input_shape = y.shape[..^1];
            var categorical = np.zeros((y.size, num_classes), dtype: dtype.as_numpy_dtype());
            // categorical[np.arange(y.size), y] = 1;
            for (int i = 0; i < y.size; i++)
            {
                categorical[i][y1[i]] = 1;
            }

            return categorical;
        }
    }
}

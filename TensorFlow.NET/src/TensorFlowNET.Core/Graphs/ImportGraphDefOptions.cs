﻿/*****************************************************************************
   Copyright 2018 The TensorFlow.NET Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/

using System;

namespace Tensorflow
{
    public sealed class ImportGraphDefOptions : IDisposable
    {
        public SafeImportGraphDefOptionsHandle Handle { get; }

        public int NumReturnOutputs
            => c_api.TF_ImportGraphDefOptionsNumReturnOutputs(Handle);

        public ImportGraphDefOptions()
        {
            Handle = c_api.TF_NewImportGraphDefOptions();
        }

        public void AddReturnOutput(string name, int index)
        {
            c_api.TF_ImportGraphDefOptionsAddReturnOutput(Handle, name, index);
        }

        public void Dispose()
            => Handle.Dispose();
    }
}

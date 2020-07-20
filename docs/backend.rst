..
  # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
  #
  # Redistribution and use in source and binary forms, with or without
  # modification, are permitted provided that the following conditions
  # are met:
  #  * Redistributions of source code must retain the above copyright
  #    notice, this list of conditions and the following disclaimer.
  #  * Redistributions in binary form must reproduce the above copyright
  #    notice, this list of conditions and the following disclaimer in the
  #    documentation and/or other materials provided with the distribution.
  #  * Neither the name of NVIDIA CORPORATION nor the names of its
  #    contributors may be used to endorse or promote products derived
  #    from this software without specific prior written permission.
  #
  # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
  # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
  # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
  # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

.. _section-backends:

Backends
========

A model using a custom backend is represented in the model repository
in the same way as models using a deep-learning framework backend.
Each model version sub-directory must contain at least one shared
library that implements the custom model backend. By default, the name
of this shared library must be **libcustom.so** but the default name
can be overridden using the *default_model_filename* property in the
:ref:`model configuration <section-model-configuration>`.

Optionally, a model can provide multiple shared libraries, each
targeted at a GPU with a different `Compute Capability
<https://developer.nvidia.com/cuda-gpus>`_. See the
*cc_model_filenames* property in the :ref:`model configuration
<section-model-configuration>` for description of how to specify
different shared libraries for different compute capabilities.

Currently, only model repositories on the local filesystem support
custom backends. A custom backend contained in a model repository in
cloud storage (for example, a repository accessed with the gs://
prefix or s3:// prefix as described above) cannot be loaded.

Custom Backend API
^^^^^^^^^^^^^^^^^^

A custom backend must implement the C interface defined in `custom.h
<https://github.com/NVIDIA/triton-inference-server/blob/master/src/backends/custom/custom.h>`_. The
interface is also documented in the API Reference.

Example Custom Backend
^^^^^^^^^^^^^^^^^^^^^^

Several example custom backends can be found in the `src/custom
directory
<https://github.com/NVIDIA/triton-inference-server/tree/master/src/custom>`_. For
more information on building your own custom backends as well as a
simple example you can build yourself, see
:ref:`section-building-a-custom-backend`.

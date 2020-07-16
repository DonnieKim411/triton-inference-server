// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/backends/backend/examples/backend_utils.h"
#include "src/backends/backend/tensorflow/model_instance.h"
#include "src/backends/backend/tensorflow/tf_utils.h"
#include "src/backends/tensorflow/tensorflow_backend_tf.h"

// FIXME move constants out
#include "src/core/constants.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <set>
#include <thread>
#include <unordered_map>
#include <sys/stat.h>
#include <dirent.h>

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace ni = nvidia::inferenceserver;
namespace nib = nvidia::inferenceserver::backend;

//
// WIP: TF Graphdef Backend that implements the TRITONBACKEND API.
//

namespace {

using IONameMap = std::unordered_map<std::string, std::string>;
using TRTISTFModelHandle =
    std::unique_ptr<TRTISTF_Model, decltype(&TRTISTF_ModelDelete)>;

#define RESPOND_AND_RETURN_IF_ERROR(REQUEST, X)                         \
  do {                                                                  \
    TRITONSERVER_Error* rarie_err__ = (X);                              \
    if (rarie_err__ != nullptr) {                                       \
      TRITONBACKEND_Response* rarie_response__ = nullptr;               \
      LOG_IF_ERROR(                                                     \
          TRITONBACKEND_ResponseNew(&rarie_response__, REQUEST),        \
          "failed to create response");                                 \
      if (rarie_response__ != nullptr) {                                \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                rarie_response__, TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                rarie_err__),                                           \
            "failed to send error response");                           \
      }                                                                 \
      TRITONSERVER_ErrorDelete(rarie_err__);                            \
      return;                                                           \
    }                                                                   \
  } while (false)

#define RESPOND_FACTORY_AND_RETURN_IF_ERROR(FACTORY, X)                      \
  do {                                                                       \
    TRITONSERVER_Error* rfarie_err__ = (X);                                  \
    if (rfarie_err__ != nullptr) {                                           \
      TRITONBACKEND_Response* rfarie_response__ = nullptr;                   \
      LOG_IF_ERROR(                                                          \
          TRITONBACKEND_ResponseNewFromFactory(&rfarie_response__, FACTORY), \
          "failed to create response");                                      \
      if (rfarie_response__ != nullptr) {                                    \
        LOG_IF_ERROR(                                                        \
            TRITONBACKEND_ResponseSend(                                      \
                rfarie_response__, TRITONSERVER_RESPONSE_COMPLETE_FINAL,     \
                rfarie_err__),                                               \
            "failed to send error response");                                \
      }                                                                      \
      TRITONSERVER_ErrorDelete(rfarie_err__);                                \
      return;                                                                \
    }                                                                        \
  } while (false)

TRITONSERVER_Error* ModelPaths(const std::string& path, const bool is_graphdef, std::unordered_map<std::string, std::string>* model_paths)
{
  std::set<std::string> model_files;
  // Read all the files in 'path' and filter by type for different platform
  RETURN_IF_ERROR(GetDirectoryContents(path, &model_files));
  if (is_graphdef) {
    // Erase directory entries...
    for (auto iter = model_files.begin(); iter != model_files.end();) {
      bool is_dir;
      RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
      if (is_dir) {
        iter = model_files.erase(iter);
      } else {
        ++iter;
      }
    }
  } else {
    // Erase non-directory entries...
    for (auto iter = model_files.begin(); iter != model_files.end();) {
      bool is_dir;
      RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
      if (!is_dir) {
        iter = model_files.erase(iter);
      } else {
        ++iter;
      }
    }
  }

  for (const auto& filename : model_files) {
    const auto model_path = JoinPath({path, filename});
    model_paths->emplace(
        std::piecewise_construct, std::make_tuple(filename),
        std::make_tuple(model_path));
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
GetDirectoryContents(
    const std::string& path, std::set<std::string>* contents)
{
  DIR* dir = opendir(path.c_str());
  if (dir == nullptr) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (std::string("failed to open directory ") + path).c_str());
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string entryname = entry->d_name;
    if ((entryname != ".") && (entryname != "..")) {
      contents->insert(entryname);
    }
  }

  closedir(dir);

  return nullptr;  // success
}

TRITONSERVER_Error*
IsDirectory(const std::string& path, bool* is_dir)
{
  *is_dir = false;

  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (std::string("failed to stat file ") + path).c_str());
  }

  *is_dir = S_ISDIR(st.st_mode);
  return nullptr;  // success
}

std::string
JoinPath(std::initializer_list<std::string> segments)
{
  std::string joined;

  for (const auto& seg : segments) {
    if (joined.empty()) {
      joined = seg;
    } else if (!seg.empty() && (seg[0] == '/')) { // IsAbsolutePath(seg)
      if (joined[joined.size() - 1] == '/') {
        joined.append(seg.substr(1));
      } else {
        joined.append(seg);
      }
    } else {  // !IsAbsolutePath(seg)
      if (joined[joined.size() - 1] != '/') {
        joined.append("/");
      }
      joined.append(seg);
    }
  }

  return joined;
}

TRITONSERVER_Error*
ParseLongLongParameter(
    const std::string& key, const std::string& value, int64_t* parsed_value)
{
  try {
    *parsed_value = std::stoll(value);
  }
  catch (const std::invalid_argument& ia) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("failed to convert ") + key + " '" + value + "' to integral number").c_str());
  }

  return nullptr;  // success
}

namespace SavedModel {
TRITONSERVER_Error*
CreateTRTISTFModel(
    ni::TritonJson::Value& backend_config,
    ni::TritonJson::Value& model_config, const int device_id,
    const bool has_graph_level, const int graph_level,
    const std::string& model_name, const std::string& model_path,
    TRTISTFModelHandle* trtistf_model, IONameMap* input_name_map,
    IONameMap* output_name_map, const TRTISTF_TFTRTConfig* tftrt_config,
    const bool auto_mixed_precision)
{
  TRTISTF_Model* model = nullptr;
  RETURN_IF_TRTISTF_ERROR(TRTISTF_ModelCreateFromSavedModel(
      &model, model_name.c_str(), model_path.c_str(), device_id,
      has_graph_level, graph_level, true,
      0, true, std::map<int, std::vector<float>>(),
      tftrt_config, auto_mixed_precision));

  trtistf_model->reset(model);

  // The model inputs are the expected inputs and the outputs are
  // the allowed outputs. Saved-model gives these explicitly so we can
  // check precisely if the model configuration matches.
  const TRTISTF_IOList* inputs = TRTISTF_ModelInputs(model);
  const TRTISTF_IOList* outputs = TRTISTF_ModelOutputs(model);

  std::set<std::string> expected_inputs, allowed_outputs;
  for (const TRTISTF_IOList* itr = inputs; itr != nullptr; itr = itr->next_) {
    expected_inputs.insert(itr->io_->name_);
    input_name_map->insert({itr->io_->name_, itr->io_->inmodel_name_});
  }
  for (const TRTISTF_IOList* itr = outputs; itr != nullptr; itr = itr->next_) {
    allowed_outputs.insert(itr->io_->name_);
    output_name_map->insert({itr->io_->name_, itr->io_->inmodel_name_});
  }

  ni::TritonJson::Value config_inputs;
  RETURN_IF_ERROR(model_config.MemberAsArray("input", &config_inputs));
  size_t expected_input_cnt = config_inputs.ArraySize();

  // TODO merge this validation with the one below
  // If this is a sequence model then make sure that the required
  // inputs are present in the model and have the correct shape and
  // datatype.
  ni::TritonJson::Value sequence_batching;
  if (model_config.Find("sequence_batching", &sequence_batching)) {
    bool have_start, have_end, have_ready, have_corrid;
    RETURN_IF_ERROR(ValidateSequenceControl(
        model_name, model_config,
        "CONTROL_SEQUENCE_START", inputs,
        false /* required */, true /* is_boolean */, &have_start));
    RETURN_IF_ERROR(ValidateSequenceControl(
        model_name, model_config,
        "CONTROL_SEQUENCE_END", inputs,
        false /* required */, true /* is_boolean */, &have_end));
    RETURN_IF_ERROR(ValidateSequenceControl(
        model_name, model_config,
        "CONTROL_SEQUENCE_READY", inputs,
        false /* required */, true /* is_boolean */, &have_ready));
    RETURN_IF_ERROR(ValidateSequenceControl(
        model_name, model_config,
        "CONTROL_SEQUENCE_CORRID", inputs,
        false /* required */, false /* is_boolean */, &have_corrid));
    if (have_start) {
      expected_input_cnt += 1;
    }
    if (have_end) {
      expected_input_cnt += 1;
    }
    if (have_ready) {
      expected_input_cnt += 1;
    }
    if (have_corrid) {
      expected_input_cnt += 1;
    }
  }

  // Verify that the model configuration input and outputs match what
  // is expected by the model.
  if (expected_inputs.size() != expected_input_cnt) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("unable to load model '" + model_name +
                                       "', configuration expects " +
                                       std::to_string(config_inputs.ArraySize()) +
                                       " inputs, model provides " +
                                       std::to_string(expected_inputs.size())).c_str());
  }

  // WIP below
  for (size_t i = 0; i < config_inputs.ArraySize(); i++) {
    ni::TritonJson::Value io;
    RETURN_IF_ERROR(config_inputs.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    // Check datatypes
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    RETURN_ERROR_IF_FALSE(ConvertDataType(io_dtype) ==
        TRTISTF_DataType::TRTISTF_TYPE_INVALID, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("unsupported datatype '") + io_dtype + "' for tensor '"
        + io_name + "' for model '" + name_ + "'");
  }
  for (const auto& io : Config().input()) {
    RETURN_IF_ERROR(CheckAllowedModelInput(io, expected_inputs));

    const TRTISTF_IO* input = FindIOByName(inputs, io.name());
    if (input == nullptr) {
      return Status(
          Status::Code::INTERNAL,
          "unexpected inference input '" + io.name() + "'");
    }

    // If a reshape is provided for the input then use that when
    // validating that the TF model matches what is expected.
    const DimsList& dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();

    if (input->shape_->rank_ != 0) {
      RETURN_IF_ERROR(CompareDims(
          Name(), io.name(), input->shape_, dims, Config().max_batch_size() > 0,
          false /* compare_exact */));
    } else {
      // The savedmodel doesn't specify a shape for the input so use the shape
      // from the model configuration
      bool supports_batching = Config().max_batch_size() > 0;
      input->shape_->rank_ =
          (size_t)(dims.size() + (supports_batching ? 1 : 0));
      input->shape_->dims_ =
          (int64_t*)malloc(input->shape_->rank_ * sizeof(int64_t));
      for (int i = 0; i < dims.size(); ++i) {
        input->shape_->dims_[i + (supports_batching ? 1 : 0)] = dims[i];
      }
    }

    if (!CompareDataType(input->data_type_, io.data_type())) {
      return Status(
          Status::Code::INVALID_ARG,
          "unable to load model '" + Name() + "', input '" + io.name() +
              "' data-type " +
              DataType_Name(ConvertDataType(input->data_type_)) +
              " doesn't match configuration data-type " +
              DataType_Name(io.data_type()));
    }
  }

  for (const auto& io : Config().output()) {
    RETURN_IF_ERROR(CheckAllowedModelOutput(io, allowed_outputs));

    const TRTISTF_IO* output = FindIOByName(outputs, io.name());
    if (output == nullptr) {
      return Status(
          Status::Code::INTERNAL,
          "unexpected inference output '" + io.name() + "'");
    }

    // If a reshape is provided for the output then use that when
    // validating that the TF model matches what is expected.
    const DimsList& dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();

    if (output->shape_->rank_ != 0) {
      RETURN_IF_ERROR(CompareDims(
          Name(), io.name(), output->shape_, dims,
          Config().max_batch_size() > 0, true /* compare_exact */));
    } else {
      // The savedmodel doesn't specify a shape for the output so use the shape
      // from the model configuration
      bool supports_batching = Config().max_batch_size() > 0;
      output->shape_->rank_ =
          (size_t)(dims.size() + (supports_batching ? 1 : 0));
      output->shape_->dims_ =
          (int64_t*)malloc(output->shape_->rank_ * sizeof(int64_t));
      for (int i = 0; i < dims.size(); ++i) {
        output->shape_->dims_[i + (supports_batching ? 1 : 0)] = dims[i];
      }
    }

    if (!CompareDataType(output->data_type_, io.data_type())) {
      return Status(
          Status::Code::INVALID_ARG,
          "unable to load model '" + Name() + "', output '" + io.name() +
              "' data-type " +
              DataType_Name(ConvertDataType(output->data_type_)) +
              " doesn't match configuration data-type " +
              DataType_Name(io.data_type()));
    }
  }

  return Status::Success;
}

TRITONSERVER_Error* ValidateSequenceControl(
    const std::string& model_name,
    ni::TritonJson::Value& model_config,
    const std::string& control_kind,
    const TRTISTF_IOList* inputs, bool required, bool is_boolean,
    bool* have_control)
{
  std::string tensor_name;
  std::string tensor_datatype;
  ni::TritonJson::Value sequence_batching;
  RETURN_IF_ERROR(model_config.MemberAsObject("sequence_batching", &sequence_batching));
  if (is_boolean) {
    RETURN_IF_ERROR(nib::GetBooleanSequenceControlProperties(
        sequence_batching, model_name, control_kind, required,
        &tensor_name, &tensor_datatype, nullptr, nullptr, nullptr, nullptr));
  } else {
    RETURN_IF_ERROR(nib::GetTypedSequenceControlProperties(
        sequence_batching, model_name, control_kind, required,
        &tensor_name, &tensor_datatype));
  }
  
  *have_control = !tensor_name.empty();
  if (*have_control) {
    const TRTISTF_IO* input = nib::FindIOByName(inputs, tensor_name);
    if (input == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("configuration specified sequence control '" + tensor_name +
              "', but model does not provide that input").c_str()));
    }

    // Control tensors must have shape [1].
    std::vector<int64_t> dims{1};

    int64_t max_batch_size;
    RETURN_IF_ERROR(model_config.MemberAsInt("max_batch_size", &max_batch_size));

    auto err = nib::CompareDims(
        model_name, tensor_name, input->shape_, dims, max_batch_size > 0,
        true /* compare_exact */);
    if (err != nullptr) {
      auto detailed_err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("unable to load model '" + model_name +
                                         "', sequence control '" + tensor_name +
                                         "': " + TRITONSERVER_ErrorMessage(err)).c_str());
      TRITONSERVER_ErrorDelete(err);
      return detailed_err;
    }

    if (!nib::CompareDataType(input->data_type_, tensor_datatype)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("unable to load model '" + model_name + "', sequence control '" +
              tensor_name + "': the model expects data-type " +
              nib::ConvertDataType(input->data_type_) +
              " but the model configuration specifies data-type " +
              tensor_datatype).c_str());
    }
  }

  return nullptr;  // success
}

}

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState {
 public:
  class Instance : public nib::ModelInstance {
   public:
    // GPU device number that indicates model will be loaded on GPUs
    // as specified in model graph
    static constexpr int MODEL_DEVICE = -2;

    Instance(
      const std::string& name, const int gpu_device, const int max_batch_size,
      const bool enable_pinned_input, const bool enable_pinned_output)
      : nib::ModelInstance(name, gpu_device, max_batch_size, enable_pinned_input, enable_pinned_output),
        input_device_id_(-1)
    {
    }

    void Run(TRITONBACKEND_Model* model, TRITONBACKEND_Request** requests,
    const uint32_t request_count) override;

    // Map from configuration name for an input to tensor name for
    // that input in the model.
    IONameMap input_name_map_;

    // Map from configuration name for an output to tensor name for
    // that output in the model.
    IONameMap output_name_map_;

    // TRTISTFModel for this context.
    TRTISTFModelHandle trtistf_model_;

    // use for GPU allocator
    int input_device_id_;
  };

  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  ~ModelState();

  TRITONSERVER_Error* CreateExecutionContexts();

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Spawn a thread to produce outputs for a request. Return the
  // request wait time before it should release.
  void ProcessRequest(TRITONBACKEND_Request* request, uint32_t* wait_ms);

 private:
  ModelState(
      TRITONBACKEND_Model* triton_model, const std::string& name,
      ni::TritonJson::Value&& model_config);
  void ResponseThread(
      TRITONBACKEND_ResponseFactory* factory_ptr, const int32_t* in_buffer_ptr,
      const int32_t* delay_buffer_ptr, const uint32_t element_count);

  TRITONSERVER_Error* CreateExecutionContext(
      const std::string& instance_name, const nib::InstanceProperties& device,
      const std::unordered_map<std::string, std::string>& paths);

  TRITONBACKEND_Model* triton_model_;
  const std::string name_;
  ni::TritonJson::Value model_config_;
  std::atomic<size_t> inflight_thread_count_;
  std::vector<std::unique_ptr<Instance>> instances_;
  nib::BlockingQueue<Instance*> available_instances_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  // We can get the model configuration as a json string from
  // config_message, parse it with our favorite json parser to create
  // DOM that we can access when we need to example the
  // configuration. We use TritonJson, which is a wrapper that returns
  // nice errors (currently the underlying implementation is
  // rapidjson... but others could be added). You can use any json
  // parser you prefer.
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  ni::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  const char* name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &name));

  *state = new ModelState(triton_model, name, std::move(model_config));

  (*state)->ValidateModelConfig();
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::CreateExecutionContexts()
{
  const char* path = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelRepositoryPath(triton_model_, &path));
  std::string platform;
  RETURN_IF_ERROR(model_config_.MemberAsString("platform", &platform));
  bool is_graphdef;
  if (platform == "tensorflow_graphdef") {
    is_graphdef = true;
  } else if (platform == "tensorflow_savedmodel") {
    is_graphdef = false;
  } else {
    RETURN_ERROR_IF_FALSE(
        false, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("platform ") + platform + " not supported");
  }
  std::vector<nib::InstanceProperties> instances;
  RETURN_IF_ERROR(nib::ParseInstanceGroups(model_config_, &instances));

  const char* cname = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model_, &cname));
  const std::string name = std::string(cname);

  std::unordered_map<std::string, std::string> model_paths;
  RETURN_IF_ERROR(ModelPaths(path, is_graphdef, &model_paths));
  for (const auto& instance : instances) {
    switch (instance.kind_) {
      case nib::InstanceProperties::Kind::CPU: {
        const std::string instance_name =
            name + "_" + std::to_string(instance.id_) + "_cpu";
        RETURN_IF_ERROR(
            CreateExecutionContext(instance_name, instance, model_paths));
        break;
      }
      case nib::InstanceProperties::Kind::GPU: {
        const std::string instance_name =
            name + "_" + std::to_string(instance.id_) + "_gpu" +
            std::to_string(instance.device_id_);
        RETURN_IF_ERROR(
            CreateExecutionContext(instance_name, instance, model_paths));
        break;
      }
      case nib::InstanceProperties::Kind::MODEL: {
        const std::string instance_name =
            name + "_" + std::to_string(instance.id_) + "_model_device";
        RETURN_IF_ERROR(
            CreateExecutionContext(instance_name, instance, model_paths));
        break;
      }
      default: {
        RETURN_ERROR_IF_FALSE(
            false, TRITONSERVER_ERROR_INVALID_ARG,
            std::string("instance setting ") + instance.AsString() +
                " not supported");
        break;
      }
    }
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::CreateExecutionContext(
    const std::string& instance_name, const nib::InstanceProperties& device,
    const std::unordered_map<std::string, std::string>& paths)
{
  // For a GPU context, determine the model file to use for device
  // compute capability. CPU always uses the default model file.
  std::string cc_model_filename;
  model_config_.MemberAsString("default_model_filename", &cc_model_filename);
  int gpu_device;

  switch (device.kind_) {
    case nib::InstanceProperties::Kind::CPU: {
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
          (std::string("Creating instance ") + instance_name +
           " on CPU using " + cc_model_filename)
              .c_str());
      gpu_device = Instance::NO_GPU_DEVICE;
      break;
    }
    case nib::InstanceProperties::Kind::MODEL: {
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
          (std::string("Creating instance ") + instance_name +
           " on devices using " + cc_model_filename)
              .c_str());
      gpu_device = Instance::MODEL_DEVICE;
      break;
    }
    default: {
#ifdef TRITON_ENABLE_GPU
      cudaDeviceProp cuprops;
      cudaError_t cuerr = cudaGetDeviceProperties(&cuprops, device.device_id_);
      if (cuerr != cudaSuccess) {
        RETURN_ERROR_IF_FALSE(
            false, TRITONSERVER_ERROR_INTERNAL,
            std::string("unable to get CUDA device properties for ") + name_ +
                ": " + cudaGetErrorString(cuerr));
      }

      const std::string cc =
          std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);
      ni::TritonJson::Value cc_names;
      ni::TritonJson::Value cc_name;
      if ((model_config_.Find("cc_model_filenames", &cc_names)) &&
          (cc_names.Find(cc.c_str(), &cc_name))) {
        cc_name.AsString(&cc_model_filename);
      }

      gpu_device = device.device_id_;
      // FIXME move virtual device utils into backend
      // // Get virtual device tracker instance, and get next device id
      // if (VirtualDeviceTracker::HasVirtualDevice()) {
      //   RETURN_IF_ERROR(
      //       VirtualDeviceTracker::GetNextVirtualDevice(gpu_device,
      //       &vgpu_device));
      // }
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
          (std::string("Creating instance ") + instance_name + " on GPU " +
           std::to_string(gpu_device) + " (" + cc + ") using " +
           cc_model_filename)
              .c_str());
#else
      RETURN_ERROR_IF_FALSE(
          false, TRITONSERVER_ERROR_INTERNAL, "GPU instances not supported");
#endif  // TRITON_ENABLE_GPU
      break;
    }
  }

  const auto& gdp_itr = paths.find(cc_model_filename);
  if (gdp_itr == paths.end()) {
    RETURN_ERROR_IF_FALSE(
          false, TRITONSERVER_ERROR_INTERNAL, (std::string("unable to find model '")
          + cc_model_filename + "' for " + name_));
  }

  // Max batch size. A value of 0 in the config becomes NO_BATCHING.
  int64_t max_batch_size;
  RETURN_IF_ERROR(model_config_.MemberAsInt("max_batch_size", &max_batch_size));
  const int mbs = (max_batch_size <= 0) ? Instance::NO_BATCHING
                                                   : max_batch_size;

  // TODO put the model config related code as backend_utils
  bool pinned_input, pinned_output;
  {
    ni::TritonJson::Value optimization;
    if (model_config_.Find("optimization", &optimization)) {
      ni::TritonJson::Value pinned_memory;
      if (model_config_.Find("input_pinned_memory", &pinned_memory)) {
        RETURN_IF_ERROR(pinned_memory.MemberAsBool("enable", &pinned_input));
      }
      if (model_config_.Find("output_pinned_memory", &pinned_memory)) {
        RETURN_IF_ERROR(pinned_memory.MemberAsBool("enable", &pinned_output));
      }
    }
  }

// FIXME metric reporter requires Triton endpoint
//   std::unique_ptr<MetricModelReporter> metric_reporter;
// #ifdef TRITON_ENABLE_METRICS
//   if (Metrics::Enabled()) {
//     metric_reporter.reset(new MetricModelReporter(
//         Name(), Version(), gpu_device, Config().metric_tags()));
//   }
// #endif  // TRITON_ENABLE_METRICS

  instances_.emplace_back(new Instance(
      instance_name, gpu_device, mbs, pinned_input, pinned_output));
  auto instance = instances_.back().get();

  RETURN_IF_ERROR(instance->CreateCudaStream());

  TRTISTF_TFTRTConfig* tftrt_config_ptr = nullptr;
  TRTISTF_TFTRTConfig tftrt_config;
  bool auto_mixed_precision = false;
  // [TODO] this can be moved one level above
  {
    ni::TritonJson::Value optimization;
    if (model_config_.Find("optimization", &optimization)) {
      ni::TritonJson::Value eas;
      if (optimization.Find("execution_accelerators", &eas)) {
        // Set default values. is_dynamic_op is always true for online
        // TF-TRT.
        tftrt_config.minimum_segment_size_ = 3;
        tftrt_config.max_workspace_size_bytes_ = 1 << 30;
        tftrt_config.max_cached_engines_ = 100;
        tftrt_config.max_batch_size_ = std::max(mbs, 1);
        tftrt_config.precision_mode_ = TRTISTF_MODE_FP32;
        tftrt_config.is_dynamic_op_ = true;

        ni::TritonJson::Value cpu_eas;
        RETURN_ERROR_IF_TRUE(eas.Find("cpu_execution_accelerator", &cpu_eas) && (cpu_eas.ArraySize() != 0),
        TRITONSERVER_ERROR_INVALID_ARG, std::string("CPU Execution Accelerator is not supported in TensorFlow backend"));

        RETURN_ERROR_IF_TRUE(gpu_device == Instance::NO_GPU_DEVICE,
        TRITONSERVER_ERROR_INVALID_ARG, std::string("GPU Execution Accelerator can only be set on non-CPU backend "
              "context"));
        
        ni::TritonJson::Value gpu_eas;
        if (eas.Find("gpu_execution_accelerator", &gpu_eas)) {
          for (size_t ea_idx = 0; ea_idx < eas.ArraySize(); ea_idx++) {
            ni::TritonJson::Value ea;
            RETURN_IF_ERROR(eas.IndexAsObject(ea_idx, &ea));
            std::string name;
            RETURN_IF_ERROR(ea.MemberAsString("name", &name));
            if (name == ni::kTensorRTExecutionAccelerator) {
              // Validate and set parameters
              ni::TritonJson::Value params;
              if (ea.Find("parameters", &params)) {
                ni::TritonJson::Value param_value;
                std::string value_string;
                if (params.Find("precision_mode", &param_value)) {
                  RETURN_IF_ERROR(param_value.AsString(&value_string));
                  if (value_string == "FP32") {
                    tftrt_config.precision_mode_ = TRTISTF_MODE_FP32;
                  } else if (value_string == "FP16") {
                    tftrt_config.precision_mode_ = TRTISTF_MODE_FP16;
                  } else {
                    RETURN_ERROR_IF_FALSE(false, TRITONSERVER_ERROR_INVALID_ARG,
                    std::string("unsupported precision mode '") +
                                                      value_string +
                                                      "' is requested");
                  }
                }
                if (params.Find("minimum_segment_size", &param_value)) {
                  RETURN_IF_ERROR(param_value.AsString(&value_string));
                  RETURN_IF_ERROR(ParseLongLongParameter(
                      "minimum_segment_size", value_string,
                      &tftrt_config.minimum_segment_size_));
                }
                if (params.Find("max_workspace_size_bytes", &param_value)) {
                  RETURN_IF_ERROR(param_value.AsString(&value_string));
                  RETURN_IF_ERROR(ParseLongLongParameter(
                      "max_workspace_size_bytes", value_string,
                      &tftrt_config.max_workspace_size_bytes_));
                }
                if (params.Find("max_cached_engines", &param_value)) {
                  RETURN_IF_ERROR(param_value.AsString(&value_string));
                  RETURN_IF_ERROR(ParseLongLongParameter(
                      "max_cached_engines", value_string,
                      &tftrt_config.max_cached_engines_));
                }
              }
              tftrt_config_ptr = &tftrt_config;
              TRITONSERVER_LogMessage(
                TRITONSERVER_LOG_VERBOSE, __FILE__, __LINE__,
                (std::string("TensorRT Execution Accelerator is set for ") + instance_name)
                    .c_str());
            } else if (name == ni::kGPUIOExecutionAccelerator) {
              // GPU I/O can be set, set hint
              if ((gpu_device != Instance::NO_GPU_DEVICE) &&
                  (gpu_device != Instance::MODEL_DEVICE)) {
                // FIXME In TensorFlow, TF device (vGPU) is used for device utilities
                instance->input_device_id_ = gpu_device;
              }
            } else if (name_ == ni::kAutoMixedPrecisionExecutionAccelerator) {
              auto_mixed_precision = true;
            } else {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unknown Execution Accelerator '") +
                                                name +
                                                "' is requested").c_str());
            }
          }
        }
      }
    }
  }

  if (auto_mixed_precision && (tftrt_config_ptr != nullptr)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Auto mixed precision can not be set with TFTRT optimization");
  }

  // FIXME WIP below
  // [TODO] use framework in model config to create model for different type
  // FIXME get backend_config
  RETURN_IF_ERROR(CreateTRTISTFModel(
      backend_config_, vgpu_device, Config().optimization().has_graph(),
      Config().optimization().graph().level(), gdp_itr->first, gdp_itr->second,
      &context->trtistf_model_, &context->input_name_map_,
      &context->output_name_map_, tftrt_config_ptr, auto_mixed_precision));


  if (context->input_device_id_ != Context::MODEL_DEVICE) {
    const size_t num_inputs = Config().input_size();
    const size_t num_outputs = Config().output_size();
    std::vector<const char*> input_names, output_names;
    std::vector<TRTISTF_DataType> input_types, output_types;
    for (const auto& io : Config().input()) {
      input_names.push_back(io.name().c_str());
      input_types.push_back(ConvertDataType(io.data_type()));
    }
    for (const auto& io : Config().output()) {
      output_names.push_back(io.name().c_str());
      output_types.push_back(ConvertDataType(io.data_type()));
    }
    TRTISTF_ModelMakeCallable(
        context->trtistf_model_.get(), input_names.data(), input_types.data(),
        num_inputs, output_names.data(), output_types.data(), num_outputs);
  }

  return Status::Success;
}

ModelState::ModelState(
    TRITONBACKEND_Model* triton_model, const std::string& name,
    ni::TritonJson::Value&& model_config)
    : triton_model_(triton_model), name_(name),
      model_config_(std::move(model_config)), inflight_thread_count_(0)
{
}

ModelState::~ModelState()
{
  // Wait for all threads to exit...
  while (inflight_thread_count_ > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  ni::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  ni::TritonJson::Value ios;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    ni::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    // Check datatypes
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    RETURN_ERROR_IF_FALSE(ConvertDataType(io_dtype) ==
        TRTISTF_DataType::TRTISTF_TYPE_INVALID, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("unsupported datatype '") + io_dtype + "' for tensor '"
        + io_name + "' for model '" + name_ + "'");
  }
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    ni::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    // Check datatypes
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    RETURN_ERROR_IF_FALSE(ConvertDataType(io_dtype) ==
        TRTISTF_DataType::TRTISTF_TYPE_INVALID, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("unsupported datatype '") + io_dtype + "' for tensor '"
        + io_name + "' for model '" + name_ + "'");
  }

  return nullptr;  // success
}

}  // namespace

/////////////

extern "C" {

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelExecute is required.
TRITONSERVER_Error*
TRITONBACKEND_ModelExecute(
    TRITONBACKEND_Model* model, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  const char* model_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &model_name));

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_ModelExecute: model ") + model_name +
       " with " + std::to_string(request_count) + " requests")
          .c_str());

  // Triton only calls model execute from a single thread at a time
  // *for a given model*. But since this backend could be used by
  // multiple models the implementation needs to handle multiple
  // models executing at the same time. Good practice for this is to
  // use only function-local and model-specific state (obtained from
  // 'model'), which is what we do here.
  ModelState* state;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelState(model, reinterpret_cast<void**>(&state)));

  // This backend does not support models that support batching, so
  // 'request_count' should always be 1.
  RETURN_ERROR_IF_FALSE(
      request_count <= 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("repeat backend does not support batched request execution"));

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);
  uint32_t wait_milliseconds = 0;

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.

  // For simplicity we process each request in a separate thread.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    state->ProcessRequest(request, &wait_milliseconds);
  }

  // Wait, release, return...
  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("waiting ") + std::to_string(wait_milliseconds) +
       " ms before releasing requests")
          .c_str());

  std::this_thread::sleep_for(std::chrono::milliseconds(wait_milliseconds));

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    // Report statistics for the request. Note that there could
    // still be responses that have not yet been sent but those
    // cannot be captured in the statistics as they reflect only the
    // request object. We use the execution start/end time for
    // compute also so that the entire execution time is associated
    // with the inference computation.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelReportStatistics(
            model, request, true /* success */, TRITONBACKEND_NO_DEVICE,
            exec_start_ns, exec_start_ns, exec_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics. This backend does not support
  // batching so the total batch size is always 1.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelReportBatchStatistics(
          model, 1 /*total_batch_size*/, exec_start_ns, exec_start_ns,
          exec_end_ns, exec_end_ns),
      "failed reporting batch request statistics");

  TRITONSERVER_LogMessage(
      TRITONSERVER_LOG_INFO, __FILE__, __LINE__,
      (std::string("TRITONBACKEND_ModelExecute: model ") + model_name +
       " released " + std::to_string(request_count) + " requests")
          .c_str());

  return nullptr;  // success
}

}  // extern "C"

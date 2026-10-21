#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "accel/runtime.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

constexpr bool kAccelEpStreamAware = false;
static_assert(!kAccelEpStreamAware, "AccelEP does not create sync streams");

namespace {

#define ACCEL_EXPORT __attribute__((visibility("default")))

enum class OpKind { MatMul, Gemm, FusedMatMul };

struct StepInput {
  std::string name;
  bool constant = false;
  bool produced = false;
  bool pretransposed = false;
  bool quant = false;
  std::vector<int64_t> profile_shape;
};

struct CompiledStep {
  OpKind kind = OpKind::MatMul;
  std::vector<StepInput> inputs;
  accel::GemmConfig gemm{};
  accel::FusedMatMulConfig fused{};
  accel::Activation activation = accel::Activation::None;
  bool has_residual = false;
  StepInput residual;
  // RMSNormalization of the prenorm residual, fused into this step. The scale is a
  // compile-time upload, not a runtime input.
  bool has_rms = false;
  float rms_eps = 1e-5f;
  std::string rms_scale_name;
  // Closing residual Add of the saved prenorm sum with this step's output.
  bool has_skip = false;
  std::string output_name;
  std::vector<int64_t> output_shape;
  bool clear_trans_a = false;
  bool clear_trans_b = false;
};

struct CompiledKernel {
  std::vector<CompiledStep> steps;
  std::vector<std::string> runtime_names;
  bool leading_dynamic = false;
  std::vector<int64_t> dyn_a_tail;
  std::vector<int64_t> dyn_b_shape;
  std::vector<int64_t> dyn_y_tail;
  // Name of the prenorm Add output held on device for the closing residual.
  std::string skip_tensor;
};

struct MatrixInfo {
  OpKind kind = OpKind::MatMul;
  std::vector<std::string> names;
  std::vector<std::vector<int64_t>> shapes;
  std::string output_name;
  std::vector<int64_t> output_shape;
  accel::GemmConfig gemm{};
  accel::FusedMatMulConfig fused{};
};

struct DynInfo {
  std::vector<int64_t> a_tail;
  std::vector<int64_t> b_shape;
  std::vector<int64_t> y_tail;
  std::vector<std::string> names;
  std::string output_name;
};

class AccelEp;

class AccelEpFactory : public OrtEpFactory {
 public:
  AccelEpFactory(std::string name, const OrtApi& api, const OrtEpApi& ep_api, const OrtLogger& logger)
      : OrtEpFactory{}, api(api), ep_api(ep_api), logger(logger), name(std::move(name)) {
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetVendorId = GetVendorIdImpl;
    GetVersion = GetVersionImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
    // ORT calls this function pointer with no null check. A null out-pointer means
    // the host keeps its own CPU copies; the EP moves tensors privately.
    CreateDataTransfer = CreateDataTransferImpl;
    IsStreamAware = IsStreamAwareImpl;
  }

  const OrtApi& api;
  const OrtEpApi& ep_api;
  const OrtLogger& logger;
  std::string name;

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* self) noexcept {
    return static_cast<const AccelEpFactory*>(self)->name.c_str();
  }
  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* /*self*/) noexcept { return "Accel"; }
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* /*self*/) noexcept { return 0xACCE; }
  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* /*self*/) noexcept { return "0.1.0"; }
  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* /*self*/) noexcept { return kAccelEpStreamAware; }

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* /*self*/,
                                                        OrtDataTransferImpl** data_transfer) noexcept {
    *data_transfer = nullptr;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* self, const OrtHardwareDevice* const* devices,
                                                         size_t num_devices, OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices, size_t* num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* self, const OrtHardwareDevice* const* /*devices*/,
                                              const OrtKeyValuePairs* const* /*metadata*/, size_t /*num_devices*/,
                                              const OrtSessionOptions* /*session_options*/, const OrtLogger* logger,
                                              OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*self*/, OrtEp* ep) noexcept;
};

class AccelEp : public OrtEp {
 public:
  AccelEp(AccelEpFactory& factory, std::string ep_name, const OrtLogger& logger)
      : OrtEp{}, factory(factory), api(factory.api), ep_api(factory.ep_api), name(std::move(ep_name)), logger(logger),
        device(accel::CreateSimDevice()) {
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetCapability = GetCapabilityImpl;
    Compile = CompileImpl;
    ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;

    OrtStatus* status = api.Logger_LogMessage(&logger, ORT_LOGGING_LEVEL_INFO,
                                              "AccelEP ready: MatMul, FusedMatMul, and Gemm run on the accelerator",
                                              __FILE__, __LINE__, __FUNCTION__);
    api.ReleaseStatus(status);
  }

  AccelEpFactory& factory;
  const OrtApi& api;
  const OrtEpApi& ep_api;
  std::string name;
  const OrtLogger& logger;
  // Device is declared first so weight buffers are freed while it is still alive.
  std::unique_ptr<accel::Device> device;
  // Packed int8 weight. The float matrix is not materialized.
  struct QuantWeight {
    accel::OwnedBuffer q;
    accel::OwnedBuffer scale;
    accel::OwnedBuffer zero;
    std::vector<int64_t> shape;
    int64_t axis = -1;
    bool has_zero = false;
  };

  std::unordered_map<std::string, accel::OwnedBuffer> weights;
  std::unordered_map<std::string, accel::OwnedBuffer> transposed_weights;
  std::unordered_map<std::string, std::unique_ptr<QuantWeight>> quant_weights;
  std::unordered_map<std::string, std::unique_ptr<CompiledKernel>> kernels;

  accel::OwnedBuffer* FindWeight(const std::string& weight_name) {
    auto it = weights.find(weight_name);
    return it == weights.end() ? nullptr : &it->second;
  }

  QuantWeight* FindQuant(const std::string& weight_name) {
    auto it = quant_weights.find(weight_name);
    return it == quant_weights.end() ? nullptr : it->second.get();
  }

  accel::OwnedBuffer* FindTransposed(const std::string& weight_name) {
    auto it = transposed_weights.find(weight_name);
    return it == transposed_weights.end() ? nullptr : &it->second;
  }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* self) noexcept {
    return static_cast<const AccelEp*>(self)->name.c_str();
  }
  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* self, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;
  static OrtStatus* ORT_API_CALL CompileImpl(OrtEp* self, const OrtGraph** graphs, const OrtNode** fused_nodes,
                                             size_t count, OrtNodeComputeInfo** node_compute_infos,
                                             OrtNode** /*ep_context_nodes*/) noexcept;
  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* /*self*/, OrtNodeComputeInfo** node_compute_infos,
                                                       size_t count) noexcept;
};

struct NodeComputeInfo : OrtNodeComputeInfo {
  explicit NodeComputeInfo(AccelEp& ep) : ep(ep) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  AccelEp& ep;

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* self, OrtNodeComputeContext* context,
                                                 void** compute_state);
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* /*self*/, void* compute_state,
                                             OrtKernelContext* kernel_context);
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*self*/, void* /*compute_state*/) {}
};

OrtStatus* Fail(const OrtApi& api, const std::string& message) {
  return api.CreateStatus(ORT_EP_FAIL, message.c_str());
}

bool OnnxDomain(const std::string& domain) { return domain.empty() || domain == "ai.onnx"; }

bool KnownFusedMatMulAttr(const std::string& name) {
  return name == "transA" || name == "transB" || name == "transBatchA" || name == "transBatchB" || name == "alpha";
}

const char* OpName(OpKind kind) {
  if (kind == OpKind::Gemm) {
    return "Gemm";
  }
  if (kind == OpKind::FusedMatMul) {
    return "FusedMatMul";
  }
  return "MatMul";
}

std::vector<Ort::ConstValueInfo> LivePorts(const std::vector<Ort::ConstValueInfo>& values) {
  std::vector<Ort::ConstValueInfo> ports;
  for (const Ort::ConstValueInfo& value : values) {
    if (value == nullptr || value.GetName().empty()) {
      continue;
    }
    ports.push_back(value);
  }
  return ports;
}

bool TensorOf(const Ort::ConstValueInfo& value, ONNXTensorElementDataType type, std::vector<int64_t>* shape,
              bool allow_negative) {
  if (value == nullptr) {
    return false;
  }
  const Ort::ConstTypeInfo type_info = value.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  const Ort::ConstTensorTypeAndShapeInfo tensor = type_info.GetTensorTypeAndShapeInfo();
  if (tensor.GetElementType() != type) {
    return false;
  }
  *shape = tensor.GetShape();
  if (!allow_negative) {
    for (int64_t dim : *shape) {
      if (dim < 0) {
        return false;
      }
    }
  }
  return true;
}

bool FloatShape(const Ort::ConstValueInfo& value, std::vector<int64_t>* shape) {
  return TensorOf(value, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape, false);
}

bool AttrPresent(const Ort::ConstOpAttr& attr) { return static_cast<const OrtOpAttr*>(attr) != nullptr; }

bool ReadIntAttr(const Ort::ConstNode& node, const char* attr_name, int64_t fallback, int64_t* out) {
  Ort::ConstOpAttr attr{nullptr};
  Ort::Status status = node.GetAttributeByName(attr_name, attr);
  // Unset optional attributes return OK and a null OrtOpAttr (Node_GetAttributeByName).
  if (!status.IsOK() || !AttrPresent(attr)) {
    *out = fallback;
    return true;
  }
  return attr.GetValue(*out).IsOK();
}

bool ReadFloatAttr(const Ort::ConstNode& node, const char* attr_name, float fallback, float* out) {
  Ort::ConstOpAttr attr{nullptr};
  Ort::Status status = node.GetAttributeByName(attr_name, attr);
  if (!status.IsOK() || !AttrPresent(attr)) {
    *out = fallback;
    return true;
  }
  return attr.GetValue(*out).IsOK();
}

bool HasAttr(const Ort::ConstNode& node, const char* attr_name) {
  Ort::ConstOpAttr attr{nullptr};
  return node.GetAttributeByName(attr_name, attr).IsOK() && AttrPresent(attr);
}

std::string TrimNul(std::string value) {
  while (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

bool MatchMatrix(const Ort::ConstNode& node, MatrixInfo* info) {
  const std::string domain = node.GetDomain();
  const std::string op = node.GetOperatorType();
  const bool fused_matmul = op == "FusedMatMul";
  if (fused_matmul) {
    if (domain != "com.microsoft") {
      return false;
    }
    for (const Ort::ConstOpAttr& attr : node.GetAttributes()) {
      if (!KnownFusedMatMulAttr(attr.GetName())) {
        return false;
      }
    }
  } else if (op == "MatMul" || op == "Gemm") {
    if (!OnnxDomain(domain)) {
      return false;
    }
  } else {
    return false;
  }

  const std::vector<Ort::ConstValueInfo> inputs = LivePorts(node.GetInputs());
  const std::vector<Ort::ConstValueInfo> outputs = LivePorts(node.GetOutputs());
  if (outputs.size() != 1) {
    return false;
  }
  info->names.clear();
  info->shapes.clear();
  for (const Ort::ConstValueInfo& input : inputs) {
    std::vector<int64_t> shape;
    if (!FloatShape(input, &shape)) {
      return false;
    }
    info->names.push_back(input.GetName());
    info->shapes.push_back(std::move(shape));
  }
  if (!FloatShape(outputs[0], &info->output_shape)) {
    return false;
  }
  info->output_name = outputs[0].GetName();

  std::vector<int64_t> inferred;
  if (op == "MatMul") {
    if (info->names.size() != 2 ||
        !accel::InferMatMulShape(info->shapes[0], info->shapes[1], &inferred) || inferred != info->output_shape) {
      return false;
    }
    info->kind = OpKind::MatMul;
  } else if (op == "Gemm") {
    if (info->names.size() != 2 && info->names.size() != 3) {
      return false;
    }
    int64_t trans_a = 0;
    int64_t trans_b = 0;
    float alpha = 1.f;
    float beta = 1.f;
    if (!ReadIntAttr(node, "transA", 0, &trans_a) || !ReadIntAttr(node, "transB", 0, &trans_b) ||
        !ReadFloatAttr(node, "alpha", 1.f, &alpha) || !ReadFloatAttr(node, "beta", 1.f, &beta)) {
      return false;
    }
    if (!std::isfinite(alpha) || (trans_a != 0 && trans_a != 1) || (trans_b != 0 && trans_b != 1)) {
      return false;
    }
    const std::vector<int64_t>* bias = info->names.size() == 3 ? &info->shapes[2] : nullptr;
    if (!accel::InferGemmShape(info->shapes[0], trans_a != 0, info->shapes[1], trans_b != 0, bias, &inferred) ||
        inferred != info->output_shape) {
      return false;
    }
    info->kind = OpKind::Gemm;
    info->gemm.trans_a = trans_a != 0;
    info->gemm.trans_b = trans_b != 0;
    info->gemm.alpha = alpha;
    info->gemm.beta = beta;
  } else if (op == "FusedMatMul") {
    int64_t trans_a = 0;
    int64_t trans_b = 0;
    int64_t trans_batch_a = 0;
    int64_t trans_batch_b = 0;
    float alpha = 1.f;
    if (info->names.size() != 2 || !ReadIntAttr(node, "transA", 0, &trans_a) ||
        !ReadIntAttr(node, "transB", 0, &trans_b) || !ReadIntAttr(node, "transBatchA", 0, &trans_batch_a) ||
        !ReadIntAttr(node, "transBatchB", 0, &trans_batch_b) || !ReadFloatAttr(node, "alpha", 1.f, &alpha)) {
      return false;
    }
    if (!std::isfinite(alpha) || trans_batch_a != 0 || trans_batch_b != 0 || (trans_a != 0 && trans_a != 1) ||
        (trans_b != 0 && trans_b != 1)) {
      return false;
    }
    if (!accel::InferFusedMatMulShape(info->shapes[0], trans_a != 0, info->shapes[1], trans_b != 0, &inferred) ||
        inferred != info->output_shape) {
      return false;
    }
    info->kind = OpKind::FusedMatMul;
    info->fused.trans_a = trans_a != 0;
    info->fused.trans_b = trans_b != 0;
    info->fused.alpha = alpha;
  } else {
    return false;
  }
  return true;
}

bool MatchLeading(const Ort::ConstNode& node, DynInfo* dyn) {
  if (!OnnxDomain(node.GetDomain()) || node.GetOperatorType() != "MatMul") {
    return false;
  }
  const std::vector<Ort::ConstValueInfo> inputs = LivePorts(node.GetInputs());
  const std::vector<Ort::ConstValueInfo> outputs = LivePorts(node.GetOutputs());
  if (inputs.size() != 2 || outputs.size() != 1) {
    return false;
  }
  std::vector<int64_t> a_shape;
  std::vector<int64_t> b_shape;
  std::vector<int64_t> y_shape;
  if (!TensorOf(inputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &a_shape, true) ||
      !TensorOf(inputs[1], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &b_shape, true) ||
      !TensorOf(outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &y_shape, true)) {
    return false;
  }
  if (a_shape.size() < 2 || b_shape.size() < 2 || y_shape.empty() || !(a_shape[0] < 0 && y_shape[0] < 0)) {
    return false;
  }
  for (size_t i = 1; i < a_shape.size(); ++i) {
    if (a_shape[i] < 0) {
      return false;
    }
  }
  for (size_t i = 1; i < y_shape.size(); ++i) {
    if (y_shape[i] < 0) {
      return false;
    }
  }
  int64_t max_b = 0;
  for (int64_t dim : b_shape) {
    if (dim < 0) {
      return false;
    }
    if (dim > max_b) {
      max_b = dim;
    }
  }
  if (a_shape.back() < 0 || b_shape[b_shape.size() - 2] < 0 || a_shape.back() != b_shape[b_shape.size() - 2]) {
    return false;
  }
  if (max_b == std::numeric_limits<int64_t>::max()) {
    return false;
  }
  const int64_t probe = std::max(max_b, static_cast<int64_t>(1)) + 1;
  if (probe < 2) {
    return false;
  }
  for (int64_t dim : b_shape) {
    if (dim == probe) {
      return false;
    }
  }
  std::vector<int64_t> a_probe = a_shape;
  std::vector<int64_t> y_probe = y_shape;
  a_probe[0] = probe;
  y_probe[0] = probe;
  std::vector<int64_t> inferred;
  if (!accel::InferMatMulShape(a_probe, b_shape, &inferred) || inferred != y_probe || inferred[0] != probe) {
    return false;
  }
  dyn->a_tail.assign(a_shape.begin() + 1, a_shape.end());
  dyn->b_shape = b_shape;
  dyn->y_tail.assign(y_shape.begin() + 1, y_shape.end());
  dyn->names = {inputs[0].GetName(), inputs[1].GetName()};
  dyn->output_name = outputs[0].GetName();
  return true;
}

bool OutputName(const Ort::ConstNode& node, std::string* name) {
  const std::vector<Ort::ConstValueInfo> outputs = LivePorts(node.GetOutputs());
  if (outputs.size() != 1) {
    return false;
  }
  *name = outputs[0].GetName();
  return !name->empty();
}

bool SoleConsumer(const Ort::ConstNode& node, Ort::ConstNode* consumer) {
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.size() != 1 || outputs[0] == nullptr || outputs[0].GetName().empty() || outputs[0].IsGraphOutput()) {
    return false;
  }
  const std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = outputs[0].GetConsumers();
  if (consumers.size() != 1 || consumers[0].node == nullptr) {
    return false;
  }
  *consumer = consumers[0].node;
  return true;
}

bool FollowingActivation(const Ort::ConstNode& node, const char* op_type, Ort::ConstNode* activation) {
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.size() != 1 || outputs[0] == nullptr || outputs[0].IsGraphOutput()) {
    return false;
  }
  const std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = outputs[0].GetConsumers();
  if (consumers.size() != 1 || consumers[0].index != 0 || consumers[0].node == nullptr) {
    return false;
  }
  const Ort::ConstNode& candidate = consumers[0].node;
  if (!OnnxDomain(candidate.GetDomain()) || candidate.GetOperatorType() != op_type) {
    return false;
  }
  const std::vector<Ort::ConstValueInfo> act_inputs = candidate.GetInputs();
  const std::vector<Ort::ConstValueInfo> act_outputs = candidate.GetOutputs();
  if (act_inputs.size() != 1 || act_outputs.size() != 1 || act_inputs[0] == nullptr || act_outputs[0] == nullptr) {
    return false;
  }
  if (act_inputs[0].GetName() != outputs[0].GetName()) {
    return false;
  }
  std::vector<int64_t> produced;
  std::vector<int64_t> activated;
  if (!FloatShape(outputs[0], &produced) || !FloatShape(act_outputs[0], &activated) || produced != activated) {
    return false;
  }
  *activation = candidate;
  return true;
}

bool FollowingRelu(const Ort::ConstNode& node, Ort::ConstNode* relu) { return FollowingActivation(node, "Relu", relu); }

bool GeluEpilogueOk(const Ort::ConstNode& node) {
  if (!OnnxDomain(node.GetDomain()) || node.GetOperatorType() != "Gelu" || node.GetSinceVersion() < 20) {
    return false;
  }
  for (const Ort::ConstOpAttr& attr : node.GetAttributes()) {
    if (attr.GetName() != "approximate") {
      return false;
    }
  }
  if (!HasAttr(node, "approximate")) {
    return true;
  }
  Ort::ConstOpAttr attr{nullptr};
  std::string value;
  if (!node.GetAttributeByName("approximate", attr).IsOK() || !attr.GetValue(value).IsOK()) {
    return false;
  }
  return TrimNul(std::move(value)) == "none";
}

bool FollowingGelu(const Ort::ConstNode& node, Ort::ConstNode* gelu) {
  Ort::ConstNode candidate{nullptr};
  if (!FollowingActivation(node, "Gelu", &candidate) || !GeluEpilogueOk(candidate)) {
    return false;
  }
  *gelu = candidate;
  return true;
}

bool ApproximateIs(const Ort::ConstNode& node, const char* expected) {
  if (!HasAttr(node, "approximate")) {
    return false;
  }
  Ort::ConstOpAttr attr{nullptr};
  std::string value;
  if (!node.GetAttributeByName("approximate", attr).IsOK() || !AttrPresent(attr) || !attr.GetValue(value).IsOK()) {
    return false;
  }
  return TrimNul(std::move(value)) == expected;
}

// ORT's Gelu kernel rejects approximate values other than "none" and "tanh".
// tanh still runs on CPU after this MatMul. Any other value throws after the
// MatMul would already have uploaded weights, so the whole group stays on CPU.
bool UnsupportedGeluConsumer(const Ort::ConstNode& node) {
  Ort::ConstNode candidate{nullptr};
  if (!FollowingActivation(node, "Gelu", &candidate) || GeluEpilogueOk(candidate)) {
    return false;
  }
  return !ApproximateIs(candidate, "tanh");
}

bool OtherInputsAreConstant(const Ort::ConstNode& producer, const Ort::ConstNode& consumer) {
  std::string produced;
  if (!OutputName(producer, &produced)) {
    return false;
  }
  bool saw = false;
  for (const Ort::ConstValueInfo& input : LivePorts(consumer.GetInputs())) {
    if (input.GetName() == produced) {
      saw = true;
      continue;
    }
    if (!input.IsConstantInitializer()) {
      return false;
    }
  }
  return saw;
}

bool SameFloatShape(const Ort::ConstValueInfo& left, const Ort::ConstValueInfo& right) {
  std::vector<int64_t> a;
  std::vector<int64_t> b;
  return FloatShape(left, &a) && FloatShape(right, &b) && a == b;
}

bool BothMatMulInputsRuntime(const Ort::ConstNode& matmul) {
  int runtime = 0;
  for (const Ort::ConstValueInfo& input : LivePorts(matmul.GetInputs())) {
    if (input.IsConstantInitializer()) {
      return false;
    }
    ++runtime;
  }
  return runtime == 2;
}

bool FollowingAdd(const Ort::ConstNode& matmul, Ort::ConstNode* add) {
  if (!OnnxDomain(matmul.GetDomain()) || matmul.GetOperatorType() != "MatMul") {
    return false;
  }
  const std::vector<Ort::ConstValueInfo> outputs = matmul.GetOutputs();
  if (outputs.size() != 1 || outputs[0] == nullptr || outputs[0].IsGraphOutput()) {
    return false;
  }
  const std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = outputs[0].GetConsumers();
  if (consumers.size() != 1 || consumers[0].node == nullptr) {
    return false;
  }
  const Ort::ConstNode& candidate = consumers[0].node;
  if (!OnnxDomain(candidate.GetDomain()) || candidate.GetOperatorType() != "Add" ||
      !candidate.GetAttributes().empty()) {
    return false;
  }
  const std::vector<Ort::ConstValueInfo> add_inputs = LivePorts(candidate.GetInputs());
  const std::vector<Ort::ConstValueInfo> add_outputs = LivePorts(candidate.GetOutputs());
  if (add_inputs.size() != 2 || add_outputs.size() != 1) {
    return false;
  }
  // A projection MatMul (one constant weight) whose Add feeds two nodes stays
  // on CPU. The prenorm context MatMul reads two activations; that Add still
  // fuses when its result feeds LayerNorm and the MLP residual. A graph output
  // has no fan-out and can still be the epilogue.
  if (!add_outputs[0].IsGraphOutput() && add_outputs[0].GetConsumers().size() > 1 &&
      !BothMatMulInputsRuntime(matmul)) {
    return false;
  }
  const std::string produced = outputs[0].GetName();
  const Ort::ConstValueInfo* other = nullptr;
  int matches = 0;
  for (const Ort::ConstValueInfo& input : add_inputs) {
    if (input.GetName() == produced) {
      ++matches;
    } else {
      other = &input;
    }
  }
  if (matches != 1 || other == nullptr) {
    return false;
  }
  if (!SameFloatShape(outputs[0], *other) || !SameFloatShape(outputs[0], add_outputs[0])) {
    return false;
  }
  *add = candidate;
  return true;
}

bool ScaleIsPerTensor(const std::vector<int64_t>& scale_shape) { return scale_shape.empty(); }

bool IsFusableDq(const Ort::ConstNode& dq) {
  if (!OnnxDomain(dq.GetDomain()) || dq.GetOperatorType() != "DequantizeLinear") {
    return false;
  }
  for (const Ort::ConstOpAttr& attr : dq.GetAttributes()) {
    const std::string name = attr.GetName();
    if (name != "axis" && name != "block_size") {
      return false;
    }
  }
  int64_t block_size = 0;
  if (!ReadIntAttr(dq, "block_size", 0, &block_size) || block_size != 0) {
    return false;
  }
  const std::vector<Ort::ConstValueInfo> inputs = LivePorts(dq.GetInputs());
  const std::vector<Ort::ConstValueInfo> outputs = LivePorts(dq.GetOutputs());
  if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1) {
    return false;
  }
  std::vector<int64_t> weight_shape;
  std::vector<int64_t> scale_shape;
  std::vector<int64_t> out_shape;
  if (!TensorOf(inputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, &weight_shape, false) || weight_shape.size() < 2 ||
      !inputs[0].IsConstantInitializer()) {
    return false;
  }
  if (!TensorOf(inputs[1], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &scale_shape, false) ||
      !inputs[1].IsConstantInitializer()) {
    return false;
  }
  if (!(ScaleIsPerTensor(scale_shape) || scale_shape.size() == 1)) {
    return false;
  }
  if (scale_shape.size() == 1 && !HasAttr(dq, "axis")) {
    return false;
  }
  if (inputs.size() == 3) {
    std::vector<int64_t> zp_shape;
    if (!TensorOf(inputs[2], ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, &zp_shape, false) || zp_shape != scale_shape ||
        !inputs[2].IsConstantInitializer()) {
      return false;
    }
  }
  if (!FloatShape(outputs[0], &out_shape) || out_shape != weight_shape) {
    return false;
  }
  if (HasAttr(dq, "axis")) {
    int64_t axis = 0;
    if (!ReadIntAttr(dq, "axis", 0, &axis)) {
      return false;
    }
    // Graph::Resolve copies the schema default axis=1 onto the node and then
    // clears proto sync, so a missing axis and an authored axis=1 are the same
    // OrtOpAttr. A rank-1 scale with that value stays on CPU.
    if (scale_shape.size() == 1 && axis == 1) {
      return false;
    }
    if (axis < 0) {
      axis += static_cast<int64_t>(weight_shape.size());
    }
    if (axis < 0 || static_cast<size_t>(axis) >= weight_shape.size()) {
      return false;
    }
    if (scale_shape.size() == 1 && scale_shape[0] != weight_shape[static_cast<size_t>(axis)]) {
      return false;
    }
  }
  return true;
}

bool ExactlyOneFusableDq(const Ort::ConstNode& matmul, Ort::ConstNode* dq) {
  if (matmul.GetOperatorType() != "MatMul") {
    return false;
  }
  int found = 0;
  Ort::ConstNode match{nullptr};
  for (const Ort::ConstValueInfo& input : LivePorts(matmul.GetInputs())) {
    const Ort::ValueInfoConsumerProducerInfo producer = input.GetProducerNode();
    if (producer.node != nullptr && IsFusableDq(producer.node)) {
      ++found;
      match = producer.node;
    }
  }
  if (found != 1) {
    return false;
  }
  *dq = match;
  return true;
}

bool ProducerIsReshapeOrTranspose(const Ort::ConstValueInfo& input) {
  const Ort::ValueInfoConsumerProducerInfo producer = input.GetProducerNode();
  if (producer.node == nullptr) {
    return false;
  }
  const std::string op = producer.node.GetOperatorType();
  return OnnxDomain(producer.node.GetDomain()) && (op == "Reshape" || op == "Transpose");
}

bool PushActivation(const Ort::ConstNode& current, std::vector<const OrtNode*>* group, std::unordered_set<size_t>* claimed,
                    Ort::ConstNode* current_out, bool* activated) {
  Ort::ConstNode activation{nullptr};
  if (FollowingRelu(current, &activation) || FollowingGelu(current, &activation)) {
    if (claimed->count(activation.GetId()) != 0) {
      return true;
    }
    group->push_back(activation);
    claimed->insert(activation.GetId());
    *current_out = activation;
    *activated = true;
  }
  return true;
}

void RememberRuntime(std::vector<std::string>* names, const StepInput& input) {
  if (input.constant || input.produced) {
    return;
  }
  for (const std::string& existing : *names) {
    if (existing == input.name) {
      return;
    }
  }
  names->push_back(input.name);
}

OrtStatus* SaveConstant(AccelEp& ep, const Ort::ConstValueInfo& input, bool* is_constant) {
  *is_constant = false;
  if (input == nullptr || input.GetName().empty() || !input.IsConstantInitializer()) {
    return nullptr;
  }
  *is_constant = true;
  const std::string name = input.GetName();
  if (ep.FindWeight(name) != nullptr) {
    return nullptr;
  }
  Ort::ConstValue value{nullptr};
  Ort::Status status = input.GetInitializer(value);
  if (!status.IsOK()) {
    return status.release();
  }
  const auto type_shape = value.GetTensorTypeAndShapeInfo();
  if (type_shape.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Fail(ep.api, "AccelEP expected float32 initializer " + name);
  }
  const std::vector<int64_t> shape = type_shape.GetShape();
  const int64_t count = accel::ElementCount(shape);
  const float* data = count == 0 ? nullptr : value.GetTensorData<float>();
  ep.weights.emplace(name, accel::Upload(*ep.device, data, shape));
  return nullptr;
}

accel::OwnedBuffer UploadBytes(accel::Device& device, const void* data, size_t bytes) {
  accel::OwnedBuffer owned;
  owned.device = &device;
  owned.buffer = device.Allocate(bytes);
  if (bytes > 0 && data != nullptr) {
    device.CopyHostToDevice(owned.buffer, data, bytes);
  }
  return owned;
}

OrtStatus* UploadDequantized(AccelEp& ep, const Ort::ConstNode& dq, const std::string& dest_name) {
  if (ep.FindQuant(dest_name) != nullptr) {
    return nullptr;
  }
  if (!IsFusableDq(dq)) {
    return Fail(ep.api, "AccelEP DequantizeLinear is not a constant int8 weight");
  }
  const std::vector<Ort::ConstValueInfo> inputs = LivePorts(dq.GetInputs());
  Ort::ConstValue weight{nullptr};
  Ort::ConstValue scale{nullptr};
  Ort::ConstValue zero_point{nullptr};
  Ort::Status weight_status = inputs[0].GetInitializer(weight);
  Ort::Status scale_status = inputs[1].GetInitializer(scale);
  if (!weight_status.IsOK()) {
    return weight_status.release();
  }
  if (!scale_status.IsOK()) {
    return scale_status.release();
  }
  const bool has_zp = inputs.size() == 3;
  if (has_zp) {
    Ort::Status zp_status = inputs[2].GetInitializer(zero_point);
    if (!zp_status.IsOK()) {
      return zp_status.release();
    }
  }
  const std::vector<int64_t> weight_shape = weight.GetTensorTypeAndShapeInfo().GetShape();
  const std::vector<int64_t> scale_shape = scale.GetTensorTypeAndShapeInfo().GetShape();
  const int64_t count = accel::ElementCount(weight_shape);
  const int64_t scale_count = accel::ElementCount(scale_shape);
  const int8_t* weight_data = count == 0 ? nullptr : weight.GetTensorData<int8_t>();
  const float* scale_data = scale_count == 0 ? nullptr : scale.GetTensorData<float>();
  const int8_t* zp_data = has_zp && scale_count > 0 ? zero_point.GetTensorData<int8_t>() : nullptr;
  // A scalar scale is one value for the whole tensor. DequantAt reads scale[0] when axis < 0.
  int64_t axis = -1;
  if (!ScaleIsPerTensor(scale_shape)) {
    if (!HasAttr(dq, "axis") || !ReadIntAttr(dq, "axis", 0, &axis)) {
      return Fail(ep.api, "AccelEP could not read DequantizeLinear axis");
    }
    if (axis < 0) {
      axis += static_cast<int64_t>(weight_shape.size());
    }
    if (axis < 0 || static_cast<size_t>(axis) >= weight_shape.size()) {
      return Fail(ep.api, "AccelEP could not read DequantizeLinear axis");
    }
  }
  auto packed = std::make_unique<AccelEp::QuantWeight>();
  packed->shape = weight_shape;
  packed->axis = axis;
  packed->has_zero = has_zp;
  packed->q = UploadBytes(*ep.device, weight_data, static_cast<size_t>(count));
  packed->scale = UploadBytes(*ep.device, scale_data, static_cast<size_t>(scale_count) * sizeof(float));
  if (has_zp) {
    packed->zero = UploadBytes(*ep.device, zp_data, static_cast<size_t>(scale_count));
  }
  ep.quant_weights.emplace(dest_name, std::move(packed));
  return nullptr;
}

struct SubNode {
  Ort::ConstNode node;
  std::string op;
  std::string domain;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

std::unordered_map<std::string, size_t> OutputIndex(const std::vector<SubNode>& nodes) {
  std::unordered_map<std::string, size_t> index;
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (const std::string& name : nodes[i].outputs) {
      index.emplace(name, i);
    }
  }
  return index;
}

int FindConsumer(const std::vector<SubNode>& nodes, const std::string& produced, size_t self) {
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (i == self) {
      continue;
    }
    for (const std::string& input : nodes[i].inputs) {
      if (input == produced) {
        return static_cast<int>(i);
      }
    }
  }
  return -1;
}

OrtStatus* BindExternal(AccelEp& ep, const SubNode& matrix, const std::string& name, const std::vector<SubNode>& nodes,
                        StepInput* input) {
  input->name = name;
  const auto produced = OutputIndex(nodes);
  auto it = produced.find(name);
  if (it != produced.end() && nodes[it->second].op == "DequantizeLinear") {
    OrtStatus* status = UploadDequantized(ep, nodes[it->second].node, name);
    if (status != nullptr) {
      return status;
    }
    input->constant = true;
    input->quant = true;
    const AccelEp::QuantWeight* weight = ep.FindQuant(name);
    if (weight == nullptr) {
      return Fail(ep.api, "AccelEP missed a packed int8 weight");
    }
    input->profile_shape = weight->shape;
    return nullptr;
  }
  const std::vector<Ort::ConstValueInfo> ports = LivePorts(matrix.node.GetInputs());
  const Ort::ConstValueInfo* port = nullptr;
  for (const Ort::ConstValueInfo& candidate : ports) {
    if (candidate.GetName() == name) {
      port = &candidate;
      break;
    }
  }
  if (port == nullptr) {
    return Fail(ep.api, "AccelEP lost track of a compiled input");
  }
  bool constant = false;
  OrtStatus* status = SaveConstant(ep, *port, &constant);
  if (status != nullptr) {
    return status;
  }
  input->constant = constant;
  std::vector<int64_t> shape;
  if (FloatShape(*port, &shape)) {
    input->profile_shape = std::move(shape);
  }
  return nullptr;
}

OrtStatus* MaybePretranspose(AccelEp& ep, StepInput* input, bool trans) {
  if (trans && input->quant) {
    return Fail(ep.api, "AccelEP does not transpose a packed int8 weight");
  }
  if (!trans || !input->constant || input->produced) {
    return nullptr;
  }
  const accel::OwnedBuffer* weight = ep.FindWeight(input->name);
  if (weight == nullptr) {
    return Fail(ep.api, "AccelEP is missing device weight " + input->name);
  }
  if (ep.FindTransposed(input->name) == nullptr) {
    accel::OwnedBuffer transposed;
    accel::PretransposeLastTwo(*ep.device, *weight, &transposed);
    ep.transposed_weights.emplace(input->name, std::move(transposed));
  }
  input->pretransposed = true;
  return nullptr;
}

bool LastAxisOnly(const Ort::ConstNode& node, const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return false;
  }
  if (!HasAttr(node, "axis")) {
    return true;
  }
  int64_t axis = 0;
  if (!ReadIntAttr(node, "axis", -1, &axis)) {
    return false;
  }
  const int64_t rank = static_cast<int64_t>(shape.size());
  if (axis < 0) {
    axis += rank;
  }
  return axis == rank - 1;
}

// Opset-23 RMSNormalization with a constant last-axis scale, no bias, and one consumer.
bool FusableRms(const Ort::ConstNode& node) {
  if (!OnnxDomain(node.GetDomain()) || node.GetOperatorType() != "RMSNormalization" || node.GetSinceVersion() < 23) {
    return false;
  }
  for (const Ort::ConstOpAttr& attr : node.GetAttributes()) {
    const std::string name = attr.GetName();
    if (name == "axis" || name == "epsilon") {
      continue;
    }
    // Graph::Resolve copies the schema default stash_type=1 onto the node.
    // That default is the float sum. Any other stash type stays on the CPU.
    if (name == "stash_type") {
      int64_t stash = 0;
      if (!attr.GetValue(stash).IsOK() || stash != 1) {
        return false;
      }
      continue;
    }
    return false;
  }
  const std::vector<Ort::ConstValueInfo> inputs = LivePorts(node.GetInputs());
  const std::vector<Ort::ConstValueInfo> outputs = LivePorts(node.GetOutputs());
  if (inputs.size() != 2 || outputs.size() != 1) {
    return false;
  }
  if (outputs[0].IsGraphOutput() || outputs[0].GetConsumers().size() != 1) {
    return false;
  }
  std::vector<int64_t> x_shape;
  std::vector<int64_t> scale_shape;
  if (!FloatShape(inputs[0], &x_shape) || x_shape.empty() || !inputs[1].IsConstantInitializer() ||
      !FloatShape(inputs[1], &scale_shape) || scale_shape.size() != 1 || scale_shape[0] != x_shape.back()) {
    return false;
  }
  if (!LastAxisOnly(node, x_shape)) {
    return false;
  }
  float eps = 0.f;
  return ReadFloatAttr(node, "epsilon", 1e-5f, &eps);
}

bool SkipAddMatches(const SubNode& add, const std::string& skip_name, const std::string& produced_name) {
  if (!OnnxDomain(add.domain) || add.op != "Add" || !add.node.GetAttributes().empty()) {
    return false;
  }
  if (add.inputs.size() != 2 || add.outputs.size() != 1 || skip_name == produced_name) {
    return false;
  }
  const bool pair = (add.inputs[0] == skip_name && add.inputs[1] == produced_name) ||
                    (add.inputs[1] == skip_name && add.inputs[0] == produced_name);
  if (!pair) {
    return false;
  }
  const std::vector<Ort::ConstValueInfo> inputs = LivePorts(add.node.GetInputs());
  const std::vector<Ort::ConstValueInfo> outputs = LivePorts(add.node.GetOutputs());
  if (inputs.size() != 2 || outputs.size() != 1) {
    return false;
  }
  return SameFloatShape(inputs[0], inputs[1]) && SameFloatShape(inputs[0], outputs[0]);
}

// Fuse the RMS that consumes this prenorm Add, and point the chain at the next matrix.
// The scale is uploaded here and is not a runtime name. No RMS leaves the step unchanged.
OrtStatus* AttachResidualRms(AccelEp& ep, const std::vector<SubNode>& nodes, const SubNode& add, CompiledStep* step,
                             CompiledKernel* kernel, std::string* previous, int* cursor,
                             std::unordered_set<size_t>* used) {
  const std::string residual_name = add.outputs[0];
  int rms_at = -1;
  for (size_t index = 0; index < nodes.size(); ++index) {
    if (nodes[index].op != "RMSNormalization") {
      continue;
    }
    for (const std::string& input : nodes[index].inputs) {
      if (input != residual_name) {
        continue;
      }
      if (rms_at >= 0) {
        return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
      }
      rms_at = static_cast<int>(index);
    }
  }
  if (rms_at < 0) {
    return nullptr;
  }
  const SubNode& rms = nodes[static_cast<size_t>(rms_at)];
  if (!FusableRms(rms.node) || rms.outputs.size() != 1) {
    return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
  }
  std::string scale_name;
  for (const std::string& input : rms.inputs) {
    if (input != residual_name) {
      scale_name = input;
    }
  }
  if (scale_name.empty()) {
    return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
  }
  StepInput scale;
  OrtStatus* status = BindExternal(ep, rms, scale_name, nodes, &scale);
  if (status != nullptr) {
    return status;
  }
  if (!scale.constant) {
    return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
  }
  float eps = 1e-5f;
  if (!ReadFloatAttr(rms.node, "epsilon", 1e-5f, &eps)) {
    return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
  }
  const int after = FindConsumer(nodes, rms.outputs[0], static_cast<size_t>(rms_at));
  if (after < 0) {
    return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
  }
  const std::string& next_op = nodes[static_cast<size_t>(after)].op;
  if (next_op != "MatMul" && next_op != "Gemm" && next_op != "FusedMatMul") {
    return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
  }
  step->has_rms = true;
  step->rms_eps = eps;
  step->rms_scale_name = std::move(scale_name);
  step->output_name = rms.outputs[0];
  kernel->skip_tensor = residual_name;
  *previous = rms.outputs[0];
  *cursor = after;
  used->insert(static_cast<size_t>(rms_at));
  return nullptr;
}

OrtStatus* CompileGraph(AccelEp& ep, const Ort::ConstGraph& graph, std::unique_ptr<CompiledKernel>* out_kernel) {
  const std::vector<Ort::ConstNode> raw_nodes = graph.GetNodes();
  if (raw_nodes.empty()) {
    return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
  }
  std::vector<SubNode> nodes;
  nodes.reserve(raw_nodes.size());
  for (const Ort::ConstNode& node : raw_nodes) {
    SubNode parsed;
    parsed.node = node;
    parsed.op = node.GetOperatorType();
    parsed.domain = node.GetDomain();
    for (const Ort::ConstValueInfo& input : LivePorts(node.GetInputs())) {
      parsed.inputs.push_back(input.GetName());
    }
    for (const Ort::ConstValueInfo& output : LivePorts(node.GetOutputs())) {
      parsed.outputs.push_back(output.GetName());
    }
    const bool known = parsed.op == "MatMul" || parsed.op == "Gemm" || parsed.op == "FusedMatMul" ||
                       parsed.op == "Relu" || parsed.op == "Gelu" || parsed.op == "Add" ||
                       parsed.op == "DequantizeLinear" || parsed.op == "RMSNormalization";
    if (!known) {
      return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
    }
    nodes.push_back(std::move(parsed));
  }

  if (nodes.size() == 1 && nodes[0].op == "MatMul") {
    MatrixInfo info;
    DynInfo dyn;
    if (!MatchMatrix(nodes[0].node, &info) && MatchLeading(nodes[0].node, &dyn)) {
      auto kernel = std::make_unique<CompiledKernel>();
      kernel->leading_dynamic = true;
      kernel->dyn_a_tail = dyn.a_tail;
      kernel->dyn_b_shape = dyn.b_shape;
      kernel->dyn_y_tail = dyn.y_tail;
      CompiledStep step;
      step.kind = OpKind::MatMul;
      step.output_name = dyn.output_name;
      step.activation = accel::Activation::None;
      for (size_t i = 0; i < dyn.names.size(); ++i) {
        StepInput input;
        OrtStatus* status = BindExternal(ep, nodes[0], dyn.names[i], nodes, &input);
        if (status != nullptr) {
          return status;
        }
        if (!input.constant) {
          input.profile_shape = i == 0 ? std::vector<int64_t>{} : dyn.b_shape;
        }
        step.inputs.push_back(std::move(input));
      }
      RememberRuntime(&kernel->runtime_names, step.inputs[0]);
      RememberRuntime(&kernel->runtime_names, step.inputs[1]);
      kernel->steps.push_back(std::move(step));
      *out_kernel = std::move(kernel);
      return nullptr;
    }
  }

  int root = -1;
  const auto produced = OutputIndex(nodes);
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].op != "MatMul" && nodes[i].op != "Gemm" && nodes[i].op != "FusedMatMul") {
      continue;
    }
    bool consumes_internal = false;
    for (const std::string& input : nodes[i].inputs) {
      auto it = produced.find(input);
      if (it == produced.end()) {
        continue;
      }
      const std::string& op = nodes[it->second].op;
      if (op == "MatMul" || op == "Gemm" || op == "FusedMatMul" || op == "Relu" || op == "Gelu" || op == "Add" ||
          op == "RMSNormalization") {
        consumes_internal = true;
      }
    }
    if (!consumes_internal) {
      if (root != -1) {
        return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
      }
      root = static_cast<int>(i);
    }
  }
  if (root < 0) {
    return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
  }

  auto kernel = std::make_unique<CompiledKernel>();
  std::unordered_set<size_t> used;
  int cursor = root;
  std::string previous;
  int matrices = 0;
  bool activated = false;
  while (cursor >= 0) {
    SubNode& matrix = nodes[static_cast<size_t>(cursor)];
    MatrixInfo info;
    if (!MatchMatrix(matrix.node, &info)) {
      return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
    }
    used.insert(static_cast<size_t>(cursor));
    ++matrices;
    CompiledStep step;
    step.kind = info.kind;
    step.gemm = info.gemm;
    step.fused = info.fused;
    step.output_name = info.output_name;
    step.output_shape = info.output_shape;
    for (size_t i = 0; i < info.names.size(); ++i) {
      StepInput input;
      input.name = info.names[i];
      input.profile_shape = info.shapes[i];
      if (!previous.empty() && input.name == previous) {
        input.produced = true;
      } else {
        OrtStatus* status = BindExternal(ep, matrix, input.name, nodes, &input);
        if (status != nullptr) {
          return status;
        }
        if (input.profile_shape.empty()) {
          input.profile_shape = info.shapes[i];
        }
      }
      step.inputs.push_back(std::move(input));
    }
    if (step.kind == OpKind::Gemm || step.kind == OpKind::FusedMatMul) {
      const bool trans_a = step.kind == OpKind::Gemm ? step.gemm.trans_a : step.fused.trans_a;
      const bool trans_b = step.kind == OpKind::Gemm ? step.gemm.trans_b : step.fused.trans_b;
      if (step.inputs.size() >= 1) {
        OrtStatus* status = MaybePretranspose(ep, &step.inputs[0], trans_a);
        if (status != nullptr) {
          return status;
        }
        step.clear_trans_a = step.inputs[0].pretransposed;
      }
      if (step.inputs.size() >= 2) {
        OrtStatus* status = MaybePretranspose(ep, &step.inputs[1], trans_b);
        if (status != nullptr) {
          return status;
        }
        step.clear_trans_b = step.inputs[1].pretransposed;
      }
    }

    if (matrix.outputs.size() != 1) {
      return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
    }
    const int follower = FindConsumer(nodes, matrix.outputs[0], static_cast<size_t>(cursor));
    previous = matrix.outputs[0];
    cursor = -1;
    if (follower >= 0) {
      const SubNode& next = nodes[static_cast<size_t>(follower)];
      if (next.op == "Relu" || next.op == "Gelu") {
        // The fused subgraph marks its own external output as a graph output.
        // Validate the epilogue by structure. GetCapability already applied IsGraphOutput.
        if (!OnnxDomain(next.domain) || next.inputs.size() != 1 || next.outputs.size() != 1 ||
            next.inputs[0] != matrix.outputs[0]) {
          return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
        }
        std::vector<int64_t> activated_shape;
        const std::vector<Ort::ConstValueInfo> act_outputs = LivePorts(next.node.GetOutputs());
        if (act_outputs.size() != 1 || !FloatShape(act_outputs[0], &activated_shape) ||
            activated_shape != step.output_shape) {
          return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
        }
        if (next.op == "Gelu") {
          if (!GeluEpilogueOk(next.node)) {
            return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
          }
          step.activation = accel::Activation::Gelu;
        } else if (!next.node.GetAttributes().empty()) {
          return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
        } else {
          step.activation = accel::Activation::Relu;
        }
        activated = true;
        used.insert(static_cast<size_t>(follower));
        step.output_name = next.outputs[0];
        step.output_shape = activated_shape;
        previous = next.outputs[0];
        const int after = FindConsumer(nodes, next.outputs[0], static_cast<size_t>(follower));
        if (after >= 0 && (nodes[static_cast<size_t>(after)].op == "MatMul" ||
                           nodes[static_cast<size_t>(after)].op == "Gemm" ||
                           nodes[static_cast<size_t>(after)].op == "FusedMatMul")) {
          cursor = after;
        } else if (after >= 0 && !kernel->skip_tensor.empty() &&
                   SkipAddMatches(nodes[static_cast<size_t>(after)], kernel->skip_tensor, next.outputs[0])) {
          // The chain ends on an activation. The closing residual stays in this step.
          step.has_skip = true;
          step.output_name = nodes[static_cast<size_t>(after)].outputs[0];
          used.insert(static_cast<size_t>(after));
        } else if (after >= 0) {
          return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
        }
      } else if (next.op == "MatMul" || next.op == "Gemm" || next.op == "FusedMatMul") {
        cursor = follower;
      } else if (next.op == "Add") {
        if (!kernel->skip_tensor.empty() && SkipAddMatches(next, kernel->skip_tensor, matrix.outputs[0])) {
          step.has_skip = true;
          step.output_name = next.outputs[0];
          used.insert(static_cast<size_t>(follower));
        } else if (matrices != 1 || activated || step.kind != OpKind::MatMul ||
                   step.activation != accel::Activation::None || !OnnxDomain(next.domain) ||
                   !next.node.GetAttributes().empty() || next.inputs.size() != 2 || next.outputs.size() != 1) {
          return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
        } else {
        int matches = 0;
        std::string residual_name;
        for (const std::string& input_name : next.inputs) {
          if (input_name == matrix.outputs[0]) {
            ++matches;
          } else {
            residual_name = input_name;
          }
        }
        const std::vector<Ort::ConstValueInfo> add_inputs = LivePorts(next.node.GetInputs());
        const std::vector<Ort::ConstValueInfo> add_outputs = LivePorts(next.node.GetOutputs());
        const std::vector<Ort::ConstValueInfo> produced = LivePorts(matrix.node.GetOutputs());
        const Ort::ConstValueInfo* other = nullptr;
        for (const Ort::ConstValueInfo& input : add_inputs) {
          if (input.GetName() == residual_name) {
            other = &input;
          }
        }
        if (matches != 1 || residual_name.empty() || other == nullptr || add_outputs.size() != 1 ||
            produced.size() != 1 || !SameFloatShape(*other, add_outputs[0]) ||
            !SameFloatShape(produced[0], add_outputs[0])) {
          return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
        }
        StepInput residual;
        OrtStatus* status = BindExternal(ep, next, residual_name, nodes, &residual);
        if (status != nullptr) {
          return status;
        }
        if (residual.profile_shape.empty()) {
          residual.profile_shape = step.output_shape;
        }
        step.has_residual = true;
        step.residual = std::move(residual);
        step.output_name = next.outputs[0];
        used.insert(static_cast<size_t>(follower));
        OrtStatus* rms_status = AttachResidualRms(ep, nodes, next, &step, kernel.get(), &previous, &cursor, &used);
        if (rms_status != nullptr) {
          return rms_status;
        }
        }
      } else {
        return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
      }
    }
    for (const StepInput& input : step.inputs) {
      RememberRuntime(&kernel->runtime_names, input);
    }
    if (step.has_residual) {
      RememberRuntime(&kernel->runtime_names, step.residual);
    }
    kernel->steps.push_back(std::move(step));
  }

  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].op == "DequantizeLinear") {
      bool consumed = false;
      for (const CompiledStep& step : kernel->steps) {
        for (const StepInput& input : step.inputs) {
          if (input.constant && !nodes[i].outputs.empty() && input.name == nodes[i].outputs[0]) {
            consumed = true;
          }
        }
      }
      if (!consumed) {
        return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
      }
      continue;
    }
    if (used.count(i) == 0) {
      return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
    }
  }
  if (kernel->steps.empty()) {
    return Fail(ep.api, "AccelEP was asked to compile an op it did not claim");
  }
  *out_kernel = std::move(kernel);
  return nullptr;
}

bool BindConstant(AccelEp& ep, const StepInput& input, accel::Operand* operand, std::vector<int64_t>* profile_shape) {
  if (input.quant) {
    AccelEp::QuantWeight* packed = ep.FindQuant(input.name);
    if (packed == nullptr || packed->q.buffer.data == nullptr || packed->scale.buffer.data == nullptr) {
      return false;
    }
    if (packed->has_zero && packed->zero.buffer.data == nullptr) {
      return false;
    }
    // Touch the packed int8 buffer, not a dequantized float matrix.
    accel::NoteConstantTouch(packed->q.buffer.id, packed->q.buffer.data, packed->q.buffer.bytes);
    operand->quant = true;
    operand->host = nullptr;
    operand->on_device = nullptr;
    operand->shape = packed->shape;
    operand->quant_plane.q = static_cast<const int8_t*>(packed->q.buffer.data);
    operand->quant_plane.scale = static_cast<const float*>(packed->scale.buffer.data);
    operand->quant_plane.zero = packed->has_zero ? static_cast<const int8_t*>(packed->zero.buffer.data) : nullptr;
    operand->quant_plane.shape = packed->shape;
    operand->quant_plane.axis = packed->axis;
    operand->quant_plane.offset = 0;
    *profile_shape = input.profile_shape.empty() ? packed->shape : input.profile_shape;
    return true;
  }
  const accel::OwnedBuffer* weight = input.pretransposed ? ep.FindTransposed(input.name) : ep.FindWeight(input.name);
  if (weight == nullptr) {
    return false;
  }
  accel::NoteConstantTouch(weight->buffer.id, weight->buffer.data, weight->buffer.bytes);
  operand->on_device = weight;
  operand->shape = weight->shape;
  *profile_shape = input.profile_shape.empty() ? weight->shape : input.profile_shape;
  return true;
}

OrtStatus* RunKernel(AccelEp& ep, const CompiledKernel& kernel, OrtKernelContext* kernel_ctx) {
  Ort::KernelContext context(kernel_ctx);
  const size_t runtime_count = context.GetInputCount();
  if (context.GetOutputCount() != 1) {
    return Fail(ep.api, "AccelEP expected one output");
  }
  if (kernel.runtime_names.size() != runtime_count) {
    return Fail(ep.api, "AccelEP was given runtime inputs it did not compile");
  }
  std::unordered_map<std::string, accel::Operand> fed;
  for (size_t i = 0; i < kernel.runtime_names.size(); ++i) {
    Ort::ConstValue value = context.GetInput(i);
    const auto type_shape = value.GetTensorTypeAndShapeInfo();
    if (type_shape.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return Fail(ep.api, "AccelEP expected a float32 runtime input");
    }
    accel::Operand operand;
    operand.shape = type_shape.GetShape();
    operand.host = operand.shape.empty() || accel::ElementCount(operand.shape) == 0 ? nullptr
                                                                                    : value.GetTensorData<float>();
    fed.emplace(kernel.runtime_names[i], std::move(operand));
  }

  accel::OwnedBuffer retained;
  accel::OwnedBuffer skip_hold;
  std::string retained_name;
  for (size_t step_index = 0; step_index < kernel.steps.size(); ++step_index) {
    const CompiledStep& step = kernel.steps[step_index];
    const bool last = step_index + 1 == kernel.steps.size();
    std::vector<accel::Operand> operands;
    accel::ProfileRecord record;
    record.op = OpName(step.kind);
    record.activation = accel::ActivationName(step.activation);
    operands.reserve(step.inputs.size());
    for (const StepInput& input : step.inputs) {
      accel::Operand operand;
      std::vector<int64_t> shown;
      if (input.produced) {
        if (retained_name != input.name || retained.device == nullptr) {
          return Fail(ep.api, "AccelEP lost a device-resident activation");
        }
        operand.on_device = &retained;
        operand.shape = retained.shape;
        shown = retained.shape;
      } else if (input.constant) {
        if (!BindConstant(ep, input, &operand, &shown)) {
          return Fail(ep.api, input.pretransposed ? "AccelEP is missing a pretransposed weight " + input.name
                                                  : "AccelEP is missing device weight " + input.name);
        }
      } else {
        auto it = fed.find(input.name);
        if (it == fed.end()) {
          return Fail(ep.api, "AccelEP ran out of runtime inputs");
        }
        operand = it->second;
        shown = operand.shape;
      }
      record.inputs.push_back(accel::ProfileTensor{input.name, shown});
      operands.push_back(std::move(operand));
    }

    accel::Operand residual_operand;
    const accel::Operand* residual = nullptr;
    if (step.has_residual) {
      std::vector<int64_t> shown;
      if (step.residual.constant) {
        if (!BindConstant(ep, step.residual, &residual_operand, &shown)) {
          return Fail(ep.api, "AccelEP is missing device weight " + step.residual.name);
        }
      } else {
        auto it = fed.find(step.residual.name);
        if (it == fed.end()) {
          return Fail(ep.api, "AccelEP ran out of runtime inputs");
        }
        residual_operand = it->second;
        shown = residual_operand.shape;
      }
      record.inputs.push_back(accel::ProfileTensor{step.residual.name, shown});
      residual = &residual_operand;
    }

    if (kernel.leading_dynamic) {
      if (operands.size() != 2 || operands[0].shape.size() != kernel.dyn_a_tail.size() + 1 ||
          operands[0].shape[0] < 0 || operands[1].shape != kernel.dyn_b_shape) {
        return Fail(ep.api, "AccelEP runtime shapes are outside the compiled matrix contract");
      }
      for (size_t i = 0; i < kernel.dyn_a_tail.size(); ++i) {
        if (operands[0].shape[i + 1] != kernel.dyn_a_tail[i]) {
          return Fail(ep.api, "AccelEP runtime shapes are outside the compiled matrix contract");
        }
      }
    }

    std::vector<int64_t> out_shape;
    bool matched = false;
    accel::GemmConfig gemm = step.gemm;
    accel::FusedMatMulConfig fused = step.fused;
    if (step.clear_trans_a) {
      gemm.trans_a = false;
      fused.trans_a = false;
    }
    if (step.clear_trans_b) {
      gemm.trans_b = false;
      fused.trans_b = false;
    }
    if (step.kind == OpKind::MatMul) {
      matched = accel::InferMatMulShape(operands[0].shape, operands[1].shape, &out_shape);
    } else if (step.kind == OpKind::Gemm) {
      matched = accel::InferGemmShape(operands[0].shape, gemm.trans_a, operands[1].shape, gemm.trans_b,
                                      operands.size() == 3 ? &operands[2].shape : nullptr, &out_shape);
    } else {
      matched = accel::InferFusedMatMulShape(operands[0].shape, fused.trans_a, operands[1].shape, fused.trans_b,
                                             &out_shape);
    }
    if (!matched) {
      return Fail(ep.api, "AccelEP runtime shapes are outside the compiled matrix contract");
    }
    if (kernel.leading_dynamic) {
      if (out_shape.empty() || out_shape[0] != operands[0].shape[0] ||
          out_shape.size() != kernel.dyn_y_tail.size() + 1) {
        return Fail(ep.api, "AccelEP runtime shapes are outside the compiled matrix contract");
      }
      for (size_t i = 0; i < kernel.dyn_y_tail.size(); ++i) {
        if (out_shape[i + 1] != kernel.dyn_y_tail[i]) {
          return Fail(ep.api, "AccelEP runtime shapes are outside the compiled matrix contract");
        }
      }
    }

    float* out_data = nullptr;
    if (last) {
      Ort::UnownedValue output = context.GetOutput(0, out_shape);
      out_data = accel::ElementCount(out_shape) == 0 ? nullptr : output.GetTensorMutableData<float>();
    }
    record.output.name = step.output_name;
    record.output.shape = out_shape;
    const accel::AcceleratorStats before = accel::GetAcceleratorStats();
    if (step.has_rms) {
      if (operands.size() != 2) {
        return Fail(ep.api, "AccelEP lost a device-resident activation");
      }
      // Keep the pre-norm sum. The normed rows become the next matmul input. Nothing is copied back.
      accel::ExecuteMatMul(*ep.device, operands[0], operands[1], nullptr, step.activation, residual, &skip_hold);
      const accel::OwnedBuffer* scale = ep.FindWeight(step.rms_scale_name);
      if (scale == nullptr || skip_hold.shape.empty() || skip_hold.buffer.data == nullptr) {
        return Fail(ep.api, "AccelEP lost the pre-norm residual");
      }
      const int64_t dim = skip_hold.shape.back();
      const int64_t count = accel::ElementCount(skip_hold.shape);
      if (dim <= 0 || count % dim != 0) {
        return Fail(ep.api, "AccelEP RMSNorm shape is not a matrix row");
      }
      const int64_t rows = count / dim;
      accel::OwnedBuffer normed;
      normed.device = ep.device.get();
      normed.shape = skip_hold.shape;
      normed.buffer = ep.device->Allocate(static_cast<size_t>(count) * sizeof(float));
      ep.device->RmsNorm(skip_hold.buffer, scale->buffer, normed.buffer, rows, dim, step.rms_eps, 0);
      if (last && count > 0) {
        ep.device->CopyDeviceToHost(out_data, normed.buffer, static_cast<size_t>(count) * sizeof(float));
      }
      retained = std::move(normed);
    } else if (step.has_skip) {
      if (operands.size() < 2) {
        return Fail(ep.api, "AccelEP lost a device-resident activation");
      }
      accel::OwnedBuffer produced;
      if (step.kind == OpKind::MatMul) {
        accel::ExecuteMatMul(*ep.device, operands[0], operands[1], nullptr, step.activation, nullptr, &produced);
      } else if (step.kind == OpKind::Gemm) {
        const accel::Operand* bias = operands.size() == 3 ? &operands[2] : nullptr;
        accel::ExecuteGemm(*ep.device, operands[0], operands[1], bias, gemm, nullptr, step.activation, &produced);
      } else {
        accel::ExecuteFusedMatMul(*ep.device, operands[0], operands[1], fused, step.activation, nullptr, &produced);
      }
      if (produced.shape != skip_hold.shape || produced.buffer.data == nullptr || skip_hold.buffer.data == nullptr) {
        return Fail(ep.api, "AccelEP closing residual shapes differ");
      }
      const int64_t count = accel::ElementCount(produced.shape);
      ep.device->ResidualAdd(produced.buffer, skip_hold.buffer, count);
      accel::NoteAddEpilogue();
      if (last && count > 0) {
        ep.device->CopyDeviceToHost(out_data, produced.buffer, static_cast<size_t>(count) * sizeof(float));
      } else if (!last) {
        retained = std::move(produced);
      }
    } else {
      accel::OwnedBuffer* retain = last ? nullptr : &retained;
      if (step.kind == OpKind::MatMul) {
        accel::ExecuteMatMul(*ep.device, operands[0], operands[1], out_data, step.activation, residual, retain);
      } else if (step.kind == OpKind::Gemm) {
        const accel::Operand* bias = operands.size() == 3 ? &operands[2] : nullptr;
        accel::ExecuteGemm(*ep.device, operands[0], operands[1], bias, gemm, out_data, step.activation, retain);
      } else {
        accel::ExecuteFusedMatMul(*ep.device, operands[0], operands[1], fused, step.activation, out_data, retain);
      }
    }
    const accel::AcceleratorStats after = accel::GetAcceleratorStats();
    record.bytes_to_device = after.bytes_to_device - before.bytes_to_device;
    record.bytes_to_host = after.bytes_to_host - before.bytes_to_host;
    accel::AppendProfileRecord(std::move(record));
    retained_name = step.output_name;
  }
  return nullptr;
}

bool ClosingAddOf(const Ort::ConstNode& add, const std::string& skip_name, const Ort::ConstNode& produced_node) {
  if (!OnnxDomain(add.GetDomain()) || add.GetOperatorType() != "Add" || !add.GetAttributes().empty()) {
    return false;
  }
  std::string produced;
  if (!OutputName(produced_node, &produced) || produced == skip_name) {
    return false;
  }
  const std::vector<Ort::ConstValueInfo> inputs = LivePorts(add.GetInputs());
  const std::vector<Ort::ConstValueInfo> outputs = LivePorts(add.GetOutputs());
  if (inputs.size() != 2 || outputs.size() != 1) {
    return false;
  }
  bool saw_skip = false;
  bool saw_produced = false;
  for (const Ort::ConstValueInfo& input : inputs) {
    if (input.GetName() == skip_name) {
      saw_skip = true;
    } else if (input.GetName() == produced) {
      saw_produced = true;
    } else {
      return false;
    }
  }
  if (!saw_skip || !saw_produced) {
    return false;
  }
  return SameFloatShape(inputs[0], inputs[1]) && SameFloatShape(inputs[0], outputs[0]);
}

// Side list only. The caller commits it after the whole prenorm RMS chain matches.
bool TryExtendPrenormRms(const Ort::ConstNode& add, std::vector<const OrtNode*>* extra) {
  const std::vector<Ort::ConstValueInfo> add_outputs = LivePorts(add.GetOutputs());
  if (add_outputs.size() != 1 || add_outputs[0].IsGraphOutput()) {
    return false;
  }
  const std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = add_outputs[0].GetConsumers();
  Ort::ConstNode rms{nullptr};
  Ort::ConstNode closing{nullptr};
  int rms_count = 0;
  bool saw_closing = false;
  for (const Ort::ValueInfoConsumerProducerInfo& consumer : consumers) {
    if (consumer.node == nullptr) {
      return false;
    }
    if (FusableRms(consumer.node)) {
      ++rms_count;
      rms = consumer.node;
    } else if (OnnxDomain(consumer.node.GetDomain()) && consumer.node.GetOperatorType() == "Add" &&
               consumer.node.GetAttributes().empty()) {
      if (saw_closing) {
        return false;
      }
      saw_closing = true;
      closing = consumer.node;
    } else {
      return false;
    }
  }
  if (rms_count != 1) {
    return false;
  }

  std::vector<const OrtNode*> chain;
  std::unordered_set<size_t> seen;
  chain.push_back(rms);
  seen.insert(rms.GetId());
  Ort::ConstNode current = rms;
  bool got_matrix = false;
  while (true) {
    Ort::ConstNode next{nullptr};
    if (!SoleConsumer(current, &next) || seen.count(next.GetId()) != 0) {
      break;
    }
    MatrixInfo matrix;
    if (!MatchMatrix(next, &matrix) || !OtherInputsAreConstant(current, next)) {
      break;
    }
    chain.push_back(next);
    seen.insert(next.GetId());
    current = next;
    got_matrix = true;
    Ort::ConstNode activation{nullptr};
    if (FollowingRelu(current, &activation) || FollowingGelu(current, &activation)) {
      if (seen.count(activation.GetId()) != 0) {
        return false;
      }
      chain.push_back(activation);
      seen.insert(activation.GetId());
      current = activation;
    }
  }
  if (!got_matrix) {
    return false;
  }
  if (saw_closing) {
    if (seen.count(closing.GetId()) != 0 || !ClosingAddOf(closing, add_outputs[0].GetName(), current)) {
      return false;
    }
    chain.push_back(closing);
  } else {
    const std::vector<Ort::ConstValueInfo> tail = LivePorts(current.GetOutputs());
    if (tail.size() != 1 || (!tail[0].IsGraphOutput() && !tail[0].GetConsumers().empty())) {
      return false;
    }
  }
  *extra = std::move(chain);
  return true;
}

OrtStatus* AccelEp::GetCapabilityImpl(OrtEp* self, const OrtGraph* ort_graph,
                                      OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    auto* ep = static_cast<AccelEp*>(self);
    const Ort::ConstGraph graph{ort_graph};
    OrtNodeFusionOptions options{};
    options.ort_version_supported = ORT_API_VERSION;
    options.drop_constant_initializers = true;

    std::unordered_set<size_t> claimed;
    for (const Ort::ConstNode& node : graph.GetNodes()) {
      if (claimed.count(node.GetId()) != 0) {
        continue;
      }
      MatrixInfo info;
      DynInfo dyn;
      const bool is_static = MatchMatrix(node, &info);
      const bool is_dynamic = !is_static && MatchLeading(node, &dyn);
      if (!is_static && !is_dynamic) {
        continue;
      }
      if (UnsupportedGeluConsumer(node)) {
        continue;
      }
      std::vector<const OrtNode*> group;
      if (is_dynamic) {
        const std::vector<Ort::ConstValueInfo> inputs = LivePorts(node.GetInputs());
        if (!inputs.empty() && ProducerIsReshapeOrTranspose(inputs[0])) {
          continue;
        }
        group.push_back(node);
        claimed.insert(node.GetId());
      } else {
        Ort::ConstNode dq{nullptr};
        if (info.kind == OpKind::MatMul && ExactlyOneFusableDq(node, &dq) && claimed.count(dq.GetId()) == 0) {
          group.push_back(dq);
          claimed.insert(dq.GetId());
        }
        group.push_back(node);
        claimed.insert(node.GetId());
        Ort::ConstNode current = node;
        bool activated = false;
        int matrices = 1;
        if (!PushActivation(current, &group, &claimed, &current, &activated)) {
          continue;
        }
        while (true) {
          Ort::ConstNode next{nullptr};
          if (!SoleConsumer(current, &next) || claimed.count(next.GetId()) != 0) {
            break;
          }
          MatrixInfo next_info;
          if (!MatchMatrix(next, &next_info) || !OtherInputsAreConstant(current, next)) {
            break;
          }
          group.push_back(next);
          claimed.insert(next.GetId());
          current = next;
          ++matrices;
          if (!PushActivation(current, &group, &claimed, &current, &activated)) {
            break;
          }
        }
        if (matrices == 1 && !activated && info.kind == OpKind::MatMul) {
          Ort::ConstNode add{nullptr};
          if (FollowingAdd(node, &add) && claimed.count(add.GetId()) == 0) {
            std::vector<const OrtNode*> extra;
            bool extend = TryExtendPrenormRms(add, &extra);
            if (extend) {
              for (const OrtNode* extra_node : extra) {
                if (claimed.count(Ort::ConstNode{extra_node}.GetId()) != 0) {
                  extend = false;
                  break;
                }
              }
            }
            group.push_back(add);
            claimed.insert(add.GetId());
            if (extend) {
              for (const OrtNode* extra_node : extra) {
                group.push_back(extra_node);
                claimed.insert(Ort::ConstNode{extra_node}.GetId());
              }
            }
          }
        }
      }
      OrtStatus* status =
          ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info, group.data(), group.size(), &options);
      if (status != nullptr) {
        return status;
      }
    }
    return nullptr;
  } catch (const Ort::Exception& ex) {
    return Ort::Status(ex).release();
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();
  }
}

OrtStatus* AccelEp::CompileImpl(OrtEp* self, const OrtGraph** graphs, const OrtNode** fused_nodes, size_t count,
                                OrtNodeComputeInfo** node_compute_infos, OrtNode** /*ep_context_nodes*/) noexcept {
  try {
    auto* ep = static_cast<AccelEp*>(self);
    for (size_t i = 0; i < count; ++i) {
      const Ort::ConstGraph graph{graphs[i]};
      std::unique_ptr<CompiledKernel> kernel;
      OrtStatus* status = CompileGraph(*ep, graph, &kernel);
      if (status != nullptr) {
        return status;
      }
      const std::string fused_name = Ort::ConstNode{fused_nodes[i]}.GetName();
      ep->kernels.emplace(fused_name, std::move(kernel));
      node_compute_infos[i] = new NodeComputeInfo(*ep);
    }
    return nullptr;
  } catch (const Ort::Exception& ex) {
    return Ort::Status(ex).release();
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();
  }
}

void AccelEp::ReleaseNodeComputeInfosImpl(OrtEp* /*self*/, OrtNodeComputeInfo** node_compute_infos,
                                          size_t count) noexcept {
  for (size_t i = 0; i < count; ++i) {
    delete static_cast<NodeComputeInfo*>(node_compute_infos[i]);
  }
}

OrtStatus* NodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* self, OrtNodeComputeContext* context,
                                            void** compute_state) {
  auto* info = static_cast<NodeComputeInfo*>(self);
  const std::string name = info->ep.ep_api.NodeComputeContext_NodeName(context);
  auto it = info->ep.kernels.find(name);
  if (it == info->ep.kernels.end()) {
    return Fail(info->ep.api, "AccelEP has no kernel for fused node " + name);
  }
  *compute_state = it->second.get();
  return nullptr;
}

OrtStatus* NodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* self, void* compute_state,
                                        OrtKernelContext* kernel_context) {
  try {
    auto* info = static_cast<NodeComputeInfo*>(self);
    return RunKernel(info->ep, *static_cast<const CompiledKernel*>(compute_state), kernel_context);
  } catch (const Ort::Exception& ex) {
    return Ort::Status(ex).release();
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();
  }
}

OrtStatus* AccelEpFactory::GetSupportedDevicesImpl(OrtEpFactory* self, const OrtHardwareDevice* const* devices,
                                                   size_t num_devices, OrtEpDevice** ep_devices, size_t max_ep_devices,
                                                   size_t* num_ep_devices) noexcept {
  try {
    auto* factory = static_cast<AccelEpFactory*>(self);
    *num_ep_devices = 0;
    for (size_t i = 0; i < num_devices && *num_ep_devices < max_ep_devices; ++i) {
      const auto hw_type = factory->api.HardwareDevice_Type(devices[i]);
      if (hw_type != OrtHardwareDeviceType_CPU) {
        continue;
      }
      OrtEpDevice* ep_device = nullptr;
      OrtStatus* status = factory->ep_api.CreateEpDevice(factory, devices[i], nullptr, nullptr, &ep_device);
      if (status != nullptr) {
        return status;
      }
      ep_devices[(*num_ep_devices)++] = ep_device;
      break;
    }
    return nullptr;
  } catch (const Ort::Exception& ex) {
    return Ort::Status(ex).release();
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();
  }
}

OrtStatus* AccelEpFactory::CreateEpImpl(OrtEpFactory* self, const OrtHardwareDevice* const* /*devices*/,
                                        const OrtKeyValuePairs* const* /*metadata*/, size_t /*num_devices*/,
                                        const OrtSessionOptions* /*session_options*/, const OrtLogger* logger,
                                        OrtEp** ep) noexcept {
  try {
    auto* factory = static_cast<AccelEpFactory*>(self);
    const OrtLogger& session_logger = logger != nullptr ? *logger : factory->logger;
    *ep = new AccelEp(*factory, factory->name, session_logger);
    return nullptr;
  } catch (const Ort::Exception& ex) {
    return Ort::Status(ex).release();
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();
  }
}

void AccelEpFactory::ReleaseEpImpl(OrtEpFactory* /*self*/, OrtEp* ep) noexcept { delete static_cast<AccelEp*>(ep); }

}  // namespace

extern "C" {

ACCEL_EXPORT OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                                          const OrtLogger* default_logger, OrtEpFactory** factories,
                                          size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();
  Ort::InitApi(ort_api);
  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT, "AccelEP needs space for one factory");
  }
  factories[0] = new AccelEpFactory(registration_name, *ort_api, *ep_api, *default_logger);
  *num_factories = 1;
  return nullptr;
}

ACCEL_EXPORT OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<AccelEpFactory*>(factory);
  return nullptr;
}

ACCEL_EXPORT void AccelEpResetStats() { accel::ResetAcceleratorStats(); }

ACCEL_EXPORT int64_t AccelEpMatmulOps() { return accel::GetAcceleratorStats().matmul_ops; }
ACCEL_EXPORT int64_t AccelEpGemmOps() { return accel::GetAcceleratorStats().gemm_ops; }
ACCEL_EXPORT int64_t AccelEpKernelMatmuls() { return accel::GetAcceleratorStats().kernel_matmuls; }
ACCEL_EXPORT int64_t AccelEpReluEpilogues() { return accel::GetAcceleratorStats().relu_epilogues; }
ACCEL_EXPORT int64_t AccelEpGeluEpilogues() { return accel::GetAcceleratorStats().gelu_epilogues; }
ACCEL_EXPORT int64_t AccelEpAddEpilogues() { return accel::GetAcceleratorStats().add_epilogues; }
ACCEL_EXPORT int64_t AccelEpDeviceTransposes() { return accel::GetAcceleratorStats().device_transposes; }
ACCEL_EXPORT int64_t AccelEpBytesToDevice() { return accel::GetAcceleratorStats().bytes_to_device; }
ACCEL_EXPORT int64_t AccelEpBytesToHost() { return accel::GetAcceleratorStats().bytes_to_host; }
ACCEL_EXPORT int AccelEpWriteProfile(const char* path) { return accel::WriteProfile(path); }
ACCEL_EXPORT int AccelEpIsStreamAware() { return static_cast<int>(kAccelEpStreamAware); }
ACCEL_EXPORT int64_t AccelEpLiveDeviceAllocs() { return accel::LiveDeviceAllocs(); }
ACCEL_EXPORT void AccelEpClearConstantTouches() { accel::ClearConstantTouches(); }
ACCEL_EXPORT int64_t AccelEpConstantTouchCount() { return accel::ConstantTouchCount(); }
ACCEL_EXPORT uint64_t AccelEpConstantTouchId(int64_t index) { return accel::ConstantTouchId(index); }
ACCEL_EXPORT uint64_t AccelEpConstantTouchAddress(int64_t index) { return accel::ConstantTouchAddress(index); }
ACCEL_EXPORT uint64_t AccelEpConstantTouchBytes(int64_t index) { return accel::ConstantTouchBytes(index); }

}  // extern "C"

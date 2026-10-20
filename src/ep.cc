#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "accel/runtime.h"

#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

#define ACCEL_EXPORT __attribute__((visibility("default")))

enum class OpKind { MatMul, Gemm, FusedMatMul };

struct CompiledInput {
  std::string name;
  bool constant = false;
};

struct CompiledKernel {
  OpKind kind = OpKind::MatMul;
  std::vector<CompiledInput> inputs;
  accel::GemmConfig gemm{};
  accel::FusedMatMulConfig fused{};
  bool relu = false;
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
  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* /*self*/) noexcept { return false; }

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
                                              "AccelEP ready: MatMul, FusedMatMul, and Gemm run on the accelerator", __FILE__,
                                              __LINE__, __FUNCTION__);
    api.ReleaseStatus(status);
  }

  AccelEpFactory& factory;
  const OrtApi& api;
  const OrtEpApi& ep_api;
  std::string name;
  const OrtLogger& logger;
  std::unique_ptr<accel::Device> device;
  std::unordered_map<std::string, accel::OwnedBuffer> weights;
  std::unordered_map<std::string, std::unique_ptr<CompiledKernel>> kernels;

  accel::OwnedBuffer* FindWeight(const std::string& weight_name) {
    auto it = weights.find(weight_name);
    return it == weights.end() ? nullptr : &it->second;
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

bool FloatShape(const Ort::ConstValueInfo& value, std::vector<int64_t>* shape) {
  if (value == nullptr) {
    return false;
  }
  const Ort::ConstTypeInfo type_info = value.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  const Ort::ConstTensorTypeAndShapeInfo tensor = type_info.GetTensorTypeAndShapeInfo();
  if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  *shape = tensor.GetShape();
  for (int64_t dim : *shape) {
    if (dim < 0) {
      return false;
    }
  }
  return true;
}

bool ReadIntAttr(const Ort::ConstNode& node, const char* attr_name, int64_t fallback, int64_t* out) {
  Ort::ConstOpAttr attr{nullptr};
  Ort::Status status = node.GetAttributeByName(attr_name, attr);
  if (!status.IsOK()) {
    *out = fallback;
    return true;
  }
  return attr.GetValue(*out).IsOK();
}

bool ReadFloatAttr(const Ort::ConstNode& node, const char* attr_name, float fallback, float* out) {
  Ort::ConstOpAttr attr{nullptr};
  Ort::Status status = node.GetAttributeByName(attr_name, attr);
  if (!status.IsOK()) {
    *out = fallback;
    return true;
  }
  return attr.GetValue(*out).IsOK();
}

bool MatchNode(const Ort::ConstNode& node, CompiledKernel* kernel) {
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
  kernel->relu = false;

  std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  std::vector<std::string> names;
  std::vector<std::vector<int64_t>> shapes;
  names.reserve(inputs.size());
  for (const Ort::ConstValueInfo& input : inputs) {
    if (input == nullptr) {
      continue;
    }
    const std::string input_name = input.GetName();
    if (input_name.empty()) {
      continue;
    }
    std::vector<int64_t> shape;
    if (!FloatShape(input, &shape)) {
      return false;
    }
    names.push_back(input_name);
    shapes.push_back(std::move(shape));
  }

  std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.size() != 1) {
    return false;
  }
  std::vector<int64_t> output_shape;
  if (!FloatShape(outputs[0], &output_shape)) {
    return false;
  }

  std::vector<int64_t> inferred;
  kernel->inputs.clear();
  if (op == "MatMul") {
    if (names.size() != 2 || !accel::InferMatMulShape(shapes[0], shapes[1], &inferred) || inferred != output_shape) {
      return false;
    }
    kernel->kind = OpKind::MatMul;
  } else if (op == "Gemm") {
    if (names.size() != 2 && names.size() != 3) {
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
    if ((trans_a != 0 && trans_a != 1) || (trans_b != 0 && trans_b != 1)) {
      return false;
    }
    const std::vector<int64_t>* bias = names.size() == 3 ? &shapes[2] : nullptr;
    if (!accel::InferGemmShape(shapes[0], trans_a != 0, shapes[1], trans_b != 0, bias, &inferred) ||
        inferred != output_shape) {
      return false;
    }
    kernel->kind = OpKind::Gemm;
    kernel->gemm.trans_a = trans_a != 0;
    kernel->gemm.trans_b = trans_b != 0;
    kernel->gemm.alpha = alpha;
    kernel->gemm.beta = beta;
  } else if (op == "FusedMatMul") {
    int64_t trans_a = 0;
    int64_t trans_b = 0;
    int64_t trans_batch_a = 0;
    int64_t trans_batch_b = 0;
    float alpha = 1.f;
    if (names.size() != 2 || !ReadIntAttr(node, "transA", 0, &trans_a) || !ReadIntAttr(node, "transB", 0, &trans_b) ||
        !ReadIntAttr(node, "transBatchA", 0, &trans_batch_a) ||
        !ReadIntAttr(node, "transBatchB", 0, &trans_batch_b) || !ReadFloatAttr(node, "alpha", 1.f, &alpha)) {
      return false;
    }
    // Batch transposition is a different layout. Leave those nodes on the CPU EP.
    if (trans_batch_a != 0 || trans_batch_b != 0 || (trans_a != 0 && trans_a != 1) || (trans_b != 0 && trans_b != 1)) {
      return false;
    }
    if (!accel::InferFusedMatMulShape(shapes[0], trans_a != 0, shapes[1], trans_b != 0, &inferred) ||
        inferred != output_shape) {
      return false;
    }
    kernel->kind = OpKind::FusedMatMul;
    kernel->fused.trans_a = trans_a != 0;
    kernel->fused.trans_b = trans_b != 0;
    kernel->fused.alpha = alpha;
  } else {
    return false;
  }

  kernel->inputs.resize(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    kernel->inputs[i].name = std::move(names[i]);
    kernel->inputs[i].constant = false;
  }
  return true;
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

OrtStatus* RunKernel(AccelEp& ep, const CompiledKernel& kernel, OrtKernelContext* kernel_ctx) {
  Ort::KernelContext context(kernel_ctx);
  const size_t runtime_count = context.GetInputCount();
  if (context.GetOutputCount() != 1) {
    return Fail(ep.api, "AccelEP expected one output");
  }

  std::vector<accel::Operand> operands;
  operands.reserve(kernel.inputs.size());
  size_t runtime_index = 0;
  for (const CompiledInput& input : kernel.inputs) {
    accel::Operand operand;
    if (input.constant) {
      const accel::OwnedBuffer* weight = ep.FindWeight(input.name);
      if (weight == nullptr) {
        return Fail(ep.api, "AccelEP is missing device weight " + input.name);
      }
      operand.on_device = weight;
      operand.shape = weight->shape;
    } else {
      if (runtime_index >= runtime_count) {
        return Fail(ep.api, "AccelEP ran out of runtime inputs");
      }
      Ort::ConstValue value = context.GetInput(runtime_index++);
      const auto type_shape = value.GetTensorTypeAndShapeInfo();
      if (type_shape.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return Fail(ep.api, "AccelEP expected a float32 runtime input");
      }
      operand.shape = type_shape.GetShape();
      operand.host = operand.shape.empty() || accel::ElementCount(operand.shape) == 0
                         ? nullptr
                         : value.GetTensorData<float>();
    }
    operands.push_back(std::move(operand));
  }
  if (runtime_index != runtime_count) {
    return Fail(ep.api, "AccelEP was given runtime inputs it did not compile");
  }

  std::vector<int64_t> out_shape;
  bool matched = false;
  if (kernel.kind == OpKind::MatMul) {
    matched = accel::InferMatMulShape(operands[0].shape, operands[1].shape, &out_shape);
  } else if (kernel.kind == OpKind::Gemm) {
    matched = accel::InferGemmShape(operands[0].shape, kernel.gemm.trans_a, operands[1].shape, kernel.gemm.trans_b,
                                    operands.size() == 3 ? &operands[2].shape : nullptr, &out_shape);
  } else {
    matched = accel::InferFusedMatMulShape(operands[0].shape, kernel.fused.trans_a, operands[1].shape,
                                           kernel.fused.trans_b, &out_shape);
  }
  if (!matched) {
    return Fail(ep.api, "AccelEP runtime shapes are outside the compiled matrix contract");
  }

  Ort::UnownedValue output = context.GetOutput(0, out_shape);
  float* out_data = accel::ElementCount(out_shape) == 0 ? nullptr : output.GetTensorMutableData<float>();
  if (kernel.kind == OpKind::MatMul) {
    accel::ExecuteMatMul(*ep.device, operands[0], operands[1], out_data, kernel.relu);
  } else if (kernel.kind == OpKind::Gemm) {
    const accel::Operand* bias = operands.size() == 3 ? &operands[2] : nullptr;
    accel::ExecuteGemm(*ep.device, operands[0], operands[1], bias, kernel.gemm, out_data, kernel.relu);
  } else {
    accel::ExecuteFusedMatMul(*ep.device, operands[0], operands[1], kernel.fused, kernel.relu, out_data);
  }
  return nullptr;
}

bool FollowingRelu(const Ort::ConstNode& node, Ort::ConstNode* relu) {
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.size() != 1 || outputs[0] == nullptr || outputs[0].IsGraphOutput()) {
    return false;
  }
  const std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = outputs[0].GetConsumers();
  if (consumers.size() != 1 || consumers[0].index != 0 || consumers[0].node == nullptr) {
    return false;
  }
  const Ort::ConstNode& candidate = consumers[0].node;
  if (!OnnxDomain(candidate.GetDomain()) || candidate.GetOperatorType() != "Relu") {
    return false;
  }
  const std::vector<Ort::ConstValueInfo> relu_inputs = candidate.GetInputs();
  const std::vector<Ort::ConstValueInfo> relu_outputs = candidate.GetOutputs();
  if (relu_inputs.size() != 1 || relu_outputs.size() != 1 || relu_inputs[0] == nullptr || relu_outputs[0] == nullptr) {
    return false;
  }
  if (relu_inputs[0].GetName() != outputs[0].GetName()) {
    return false;
  }
  std::vector<int64_t> produced;
  std::vector<int64_t> activated;
  if (!FloatShape(outputs[0], &produced) || !FloatShape(relu_outputs[0], &activated) || produced != activated) {
    return false;
  }
  *relu = candidate;
  return true;
}

OrtStatus* AccelEp::GetCapabilityImpl(OrtEp* self, const OrtGraph* ort_graph,
                                      OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    auto* ep = static_cast<AccelEp*>(self);
    const Ort::ConstGraph graph{ort_graph};
    OrtNodeFusionOptions options{};
    options.ort_version_supported = ORT_API_VERSION;
    // Weights are copied onto the device during Compile, so ORT can drop them
    // from the fused node. Activations stay as runtime inputs.
    options.drop_constant_initializers = true;

    std::unordered_set<size_t> claimed;
    for (const Ort::ConstNode& node : graph.GetNodes()) {
      if (claimed.count(node.GetId()) != 0) {
        continue;
      }
      CompiledKernel kernel;
      if (!MatchNode(node, &kernel)) {
        continue;
      }
      // A following Relu joins this matrix op. The product stays on the device
      // and GemmEpilogue applies the activation. Every other op stays on the CPU EP.
      Ort::ConstNode relu{nullptr};
      const OrtNode* raw[2] = {node, nullptr};
      size_t count = 1;
      if (FollowingRelu(node, &relu)) {
        raw[1] = relu;
        count = 2;
        claimed.insert(relu.GetId());
      }
      claimed.insert(node.GetId());
      OrtStatus* status = ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info, raw, count, &options);
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
      const std::vector<Ort::ConstNode> nodes = graph.GetNodes();
      if (nodes.empty() || nodes.size() > 2) {
        return Fail(ep->api, "AccelEP compiles one matrix op, optionally followed by Relu");
      }
      Ort::ConstNode compute{nullptr};
      Ort::ConstNode relu{nullptr};
      for (const Ort::ConstNode& node : nodes) {
        if (OnnxDomain(node.GetDomain()) && node.GetOperatorType() == "Relu") {
          if (relu != nullptr) {
            return Fail(ep->api, "AccelEP was asked to compile more than one Relu");
          }
          relu = node;
        } else {
          if (compute != nullptr) {
            return Fail(ep->api, "AccelEP was asked to compile more than one matrix op");
          }
          compute = node;
        }
      }
      const bool grouped = (nodes.size() == 1 && relu == nullptr) || (nodes.size() == 2 && relu != nullptr);
      if (compute == nullptr || !grouped) {
        return Fail(ep->api, "AccelEP was asked to compile an op it did not claim");
      }
      auto kernel = std::make_unique<CompiledKernel>();
      if (!MatchNode(compute, kernel.get())) {
        return Fail(ep->api, "AccelEP was asked to compile an op it did not claim");
      }
      if (relu != nullptr) {
        const std::vector<Ort::ConstValueInfo> produced = compute.GetOutputs();
        const std::vector<Ort::ConstValueInfo> activated = relu.GetInputs();
        if (produced.size() != 1 || activated.size() != 1 || produced[0].GetName() != activated[0].GetName()) {
          return Fail(ep->api, "AccelEP Relu is not the consumer of the matrix op");
        }
        kernel->relu = true;
      }

      std::vector<Ort::ConstValueInfo> inputs = compute.GetInputs();
      size_t recorded = 0;
      for (const Ort::ConstValueInfo& input : inputs) {
        if (input == nullptr || input.GetName().empty()) {
          continue;
        }
        if (recorded >= kernel->inputs.size() || kernel->inputs[recorded].name != input.GetName()) {
          return Fail(ep->api, "AccelEP lost track of a compiled input");
        }
        bool is_constant = false;
        OrtStatus* status = SaveConstant(*ep, input, &is_constant);
        if (status != nullptr) {
          return status;
        }
        kernel->inputs[recorded].constant = is_constant;
        ++recorded;
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

OrtStatus* NodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* self, void* compute_state, OrtKernelContext* kernel_context) {
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
      // One CPU device is enough for this simulator.
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
ACCEL_EXPORT int64_t AccelEpBytesToDevice() { return accel::GetAcceleratorStats().bytes_to_device; }
ACCEL_EXPORT int64_t AccelEpBytesToHost() { return accel::GetAcceleratorStats().bytes_to_host; }

}  // extern "C"

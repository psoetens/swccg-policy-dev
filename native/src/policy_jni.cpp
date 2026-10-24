// In-JVM policy inference (plan .claude/plans/rollout-throughput.md, step 2): libtorch + CUDA graphs, called through JNI from
// com.gempukku.swccgo.bench.NativePolicy (Gemp harness, test scope).
//
// A slot = one TorchScript policy (swccg_policy.native: forward(tok, tokf, tmask, opt, omask, head) -> logits float32 [B, O]).
// forward() takes one padded bucket batch in a direct ByteBuffer with rl.GraphForward's layout (fields in this order, each
// starting at a 16-byte boundary): tok int16 [B,T,8], tokf float32 [B,T,4], tmask uint8 [B,T], opt int32 [B,O,6],
// omask uint8 [B,O], head int32 [B,22]; the logits [B,O] go into a direct float32 buffer. Per (B,T,O) bucket the slot keeps a
// pinned host buffer, a device buffer and a CUDA graph (captured on first use): a call = memcpy into pinned memory, one
// host-to-device copy, graph replay, one device-to-host copy, one stream synchronisation.
// swap() loads another .ts and copies its parameters and buffers into the slot's tensors in place, so the graphs stay valid.
// All GPU work of the library is serialised by one mutex (graph capture must not overlap other CUDA work of the process).
#include <jni.h>

#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/Context.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

#include <cstring>
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

std::mutex g_gpu;  // serialises every CUDA operation of this library

// No TorchScript runtime fusion (user 2026-10-08). Every rollout JVM crash so far (hs_err_pid*.log, 3-8 Oct) was a
// SIGSEGV in the policy-batcher thread inside the fuser's runtime CUDA compile (CudaCodeGen::CompileToNVRTC,
// at::cuda::jit::initializeCudaContext), 6-29 s after start, while new batch shapes are profiled and fused; models are
// loaded on other threads at the same time (torch::jit::load outside g_gpu). The forward runs as a captured CUDA graph, so
// fused kernels matter little. Set once when the library is loaded, before any model.
const bool g_no_fusion = [] {
  torch::jit::setTensorExprFuserEnabled(false);
  torch::jit::overrideCanFuseOnGPU(false);
  torch::jit::overrideCanFuseOnCPU(false);
  return true;
}();

struct Layout {
  size_t off[7];
  size_t bytes[7];
  size_t total;
};

// Plan objective-dojo Amendment 2: tensor ABI 5 = ABI 4 with eight floats per token.
static bool v4(int abi) { return abi == 4 || abi == 5; }
static int64_t floats(int abi) { return abi == 5 ? 8 : 4; }

Layout layout(int B, int T, int O, int K = 0, int abi = 3) {
  const size_t n[7] = {size_t(B) * T * 8 * (v4(abi) ? 4 : 2), size_t(B) * T * 4 * floats(abi), size_t(B) * T, size_t(B) * O * 6 * 4, size_t(B) * O,
                       size_t(B) * 22 * 4, size_t(B) * K * 153};
  Layout l{};
  size_t off = 0;
  for (int i = 0; i < (v4(abi) ? 7 : 6); i++) {
    l.off[i] = off;
    l.bytes[i] = n[i];
    off += (n[i] + 15) / 16 * 16;
  }
  l.total = off;
  return l;
}

struct Bucket {
  Layout lay;
  at::Tensor host, dev, out, out_host;
  at::Tensor sparse_host, sparse_dev;
  std::vector<at::Tensor> args;
  std::unique_ptr<at::cuda::CUDAGraph> graph;
  // completion of the last launch: a blocking-sync event, so waiting sleeps instead of spinning on the CPU
  at::cuda::CUDAEvent done{cudaEventBlockingSync | cudaEventDisableTiming};
  int B = 0, O = 0;
};

struct Slot {
  torch::jit::Module mod;
  using Key = std::tuple<int, int, int, int, int, int, int>;
  std::map<Key, std::unique_ptr<Bucket>> buckets;
  std::unordered_map<std::string, at::Tensor> tensors;  // named parameters + buffers (for swap)
  c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool();
  bool graphs = true;
  bool sparse = false;
  int abi = 1;
  std::unique_ptr<Bucket> eager;
  Key eager_key{0, 0, 0, 0, 0, 0, 0};
  at::cuda::MempoolId_t pool{0, 0};  // one memory pool shared by the slot's graphs (they never replay concurrently)
  Bucket* pending = nullptr;          // launched, not yet collected
};

// Shapes with a CUDA graph (as rl.GraphForward: larger batches / token or option counts run eagerly).
constexpr int MAX_GRAPH_B = 64, MAX_GRAPH_O = 64;
const int MAX_GRAPH_T = std::getenv("SWCCG_GRAPH_T") ? std::stoi(std::getenv("SWCCG_GRAPH_T")) : 320;
constexpr size_t MAX_GRAPH_BUCKETS = 64;

std::string g_error;

void collect(Slot& s) {
  s.tensors.clear();
  for (const auto& p : s.mod.named_parameters(true)) s.tensors[p.name] = p.value;
  for (const auto& b : s.mod.named_buffers(true)) s.tensors[b.name] = b.value;
}

std::string jstr(JNIEnv* env, jstring s) {
  const char* c = env->GetStringUTFChars(s, nullptr);
  std::string r(c);
  env->ReleaseStringUTFChars(s, c);
  return r;
}

std::vector<at::Tensor> views(const at::Tensor& dev, const Layout& l, int B, int T, int O, int K, int abi) {
  auto v = [&](int i, at::ScalarType t, std::vector<int64_t> shape) {
    return dev.narrow(0, l.off[i], l.bytes[i]).view(t).view(shape);
  };
  std::vector<at::Tensor> result{v(0, v4(abi) ? at::kInt : at::kShort, {B, T, 8}), v(1, at::kFloat, {B, T, floats(abi)}), v(2, at::kBool, {B, T}), v(3, at::kInt, {B, O, 6}),
          v(4, at::kBool, {B, O}), v(5, at::kInt, {B, 22})};
  if (v4(abi)) result.push_back(v(6, at::kByte, {B, K, 153}));
  return result;
}

at::Tensor run(Slot& s, const std::vector<at::Tensor>& args) {
  std::vector<c10::IValue> in(args.begin(), args.end());
  // Large ABI-4 option tails cannot use graphs. Use the ordinary forward here
  // so sparse-row execution does not perturb the ordering of very close logits.
  if (v4(s.abi) && args[3].size(1) > MAX_GRAPH_O) {
    in.resize(7);
    torch::jit::GraphOptimizerEnabledGuard optimizer_guard(false);
    return s.mod.forward(in).toTensor();
  }
  if (s.sparse) return s.mod.get_method("forward_packed")(in).toTensor();
  return s.mod.forward(in).toTensor();
}

struct SparseRows {
  std::vector<int64_t> rows[3];
  int capacity[3]{0, 0, 0};
};

SparseRows sparse_rows(const void* input, int B, int T, int O, bool enabled, int K = 0, int abi = 3) {
  SparseRows out;
  if (!enabled) return out;
  const auto l = layout(B, T, O, K, abi);
  const auto bytes = static_cast<const uint8_t*>(input);
  const auto tok = reinterpret_cast<const int16_t*>(bytes + l.off[0]);
  const auto tok32 = reinterpret_cast<const int32_t*>(bytes + l.off[0]);
  const auto mask = bytes + l.off[2];
  for (int64_t i = 0; i < int64_t(B) * T; ++i) {
    if (!mask[i]) continue;
    const int kind = v4(abi) ? tok32[i * 8] : tok[i * 8];
    if (v4(abi) && kind == 7 && tok32[i * 8 + 5] > 0) out.rows[0].push_back(i);
    if (kind >= 8 && kind <= 10) out.rows[kind - 8].push_back(i);
  }
  for (int i = 0; i < 3; ++i) {
    out.capacity[i] = 1;
    while (size_t(out.capacity[i]) < out.rows[i].size()) out.capacity[i] *= 2;
  }
  return out;
}

void copy_input(Bucket& b, const void* input, const SparseRows& rows) {
  std::memcpy(b.host.data_ptr(), input, b.lay.total);
  b.dev.copy_(b.host, /*non_blocking=*/true);
  if (b.sparse_host.defined()) {
    auto values = b.sparse_host.data_ptr<int64_t>();
    int offset = 0;
    for (int i = 0; i < 3; ++i) {
      std::fill(values + offset, values + offset + rows.capacity[i], int64_t(-1));
      std::copy(rows.rows[i].begin(), rows.rows[i].end(), values + offset);
      offset += rows.capacity[i];
    }
    b.sparse_dev.copy_(b.sparse_host, /*non_blocking=*/true);
  }
}

Bucket& bucket(Slot& s, int B, int T, int O, const SparseRows& rows, int K = 0) {
  auto key = std::make_tuple(B, T, O, K, rows.capacity[0], rows.capacity[1], rows.capacity[2]);
  auto it = s.buckets.find(key);
  if (it != s.buckets.end()) return *it->second;
  const bool capture = s.graphs && B <= MAX_GRAPH_B && T <= MAX_GRAPH_T && O <= MAX_GRAPH_O
      && s.buckets.size() < MAX_GRAPH_BUCKETS;
  if (!capture && s.eager && s.eager_key == key) return *s.eager;
  // A collected eager batch can be replaced; retain only one non-graph shape.
  if (!capture) s.eager.reset();
  auto b = std::make_unique<Bucket>();
  b->lay = layout(B, T, O, K, s.abi);
  b->host = at::empty({int64_t(b->lay.total)}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
  b->dev = at::zeros({int64_t(b->lay.total)}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
  b->args = views(b->dev, b->lay, B, T, O, K, s.abi);
  if (s.sparse) {
    const int total = rows.capacity[0] + rows.capacity[1] + rows.capacity[2];
    b->sparse_host = at::empty({total}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
    b->sparse_dev = at::full({total}, -1, at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
    int offset = 0;
    for (int i = 0; i < 3; ++i) {
      b->args.push_back(b->sparse_dev.narrow(0, offset, rows.capacity[i]));
      offset += rows.capacity[i];
    }
  }
  if (!v4(s.abi)) b->args[1].fill_(NAN);  // tokf padding (only matters for the warm-up / capture inputs)
  if (capture) {
    c10::cuda::CUDAStreamGuard guard(s.stream);
    for (int i = 0; i < 2; i++) run(s, b->args);  // warm-up (cuBLAS handles, allocator) outside the capture
    s.stream.synchronize();
    b->graph = std::make_unique<at::cuda::CUDAGraph>();
    b->graph->capture_begin(s.pool, cudaStreamCaptureModeThreadLocal);
    b->out = run(s, b->args);
    b->graph->capture_end();
    s.pool = b->graph->pool();
  }
  b->out_host = at::empty({B, O}, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
  if (!capture) {s.eager_key = key; s.eager = std::move(b); return *s.eager;}
  return *s.buckets.emplace(key, std::move(b)).first->second;
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL Java_com_gempukku_swccgo_bench_NativePolicy_setTf32(JNIEnv*, jclass, jboolean on) {
  at::globalContext().setAllowTF32CuBLAS(on);
  at::globalContext().setAllowTF32CuDNN(on);
}

JNIEXPORT jstring JNICALL Java_com_gempukku_swccgo_bench_NativePolicy_lastError(JNIEnv* env, jclass) {
  return env->NewStringUTF(g_error.c_str());
}

JNIEXPORT jlong JNICALL Java_com_gempukku_swccgo_bench_NativePolicy_load(JNIEnv* env, jclass, jstring path, jboolean graphs) {
  try {
    std::string p = jstr(env, path);
    auto cpu = torch::jit::load(p, at::kCPU);  // disk + deserialisation outside the GPU lock
    if (const char* expected = std::getenv("SWCCG_TENSOR_ABI")) {
      const int actual = cpu.hasattr("tensor_abi") ? cpu.attr("tensor_abi").toTensor().item<int>() : 1;
      if (actual != std::stoi(expected)) throw std::runtime_error("native tensor ABI mismatch");
    }
    std::lock_guard<std::mutex> lock(g_gpu);
    auto s = new Slot();
    s->mod = cpu;
    s->mod.to(at::kCUDA);
    s->mod.eval();
    s->abi = cpu.hasattr("tensor_abi") ? cpu.attr("tensor_abi").toTensor().item<int>() : 1;
    s->sparse = v4(s->abi) && bool(cpu.find_method("forward_packed"));
    s->graphs = graphs && (!v4(s->abi) || s->sparse);
    collect(*s);
    return reinterpret_cast<jlong>(s);
  } catch (const std::exception& e) {
    g_error = e.what();
    return 0;
  }
}

JNIEXPORT jint JNICALL Java_com_gempukku_swccgo_bench_NativePolicy_swap(JNIEnv* env, jclass, jlong h, jstring path) {
  try {
    auto& s = *reinterpret_cast<Slot*>(h);
    auto cpu = torch::jit::load(jstr(env, path), at::kCPU);
    std::unordered_map<std::string, at::Tensor> fresh;
    for (const auto& p : cpu.named_parameters(true)) fresh[p.name] = p.value;
    for (const auto& b : cpu.named_buffers(true)) fresh[b.name] = b.value;
    std::lock_guard<std::mutex> lock(g_gpu);
    if (fresh.size() != s.tensors.size()) throw std::runtime_error("swap: different tensor set");
    const int fresh_abi = cpu.hasattr("tensor_abi") ? cpu.attr("tensor_abi").toTensor().item<int>() : 1;
    if (fresh_abi != s.abi) throw std::runtime_error("swap: tensor ABI changed");
    if (v4(s.abi) && (!fresh.count("table_sha") || !at::equal(fresh.at("table_sha"), s.tensors.at("table_sha").cpu())))
      throw std::runtime_error("swap: acting table SHA changed; explicit new stage required");
    // Validate the complete update before copying any tensor.
    for (const auto& kv : s.tensors) {
      auto f = fresh.find(kv.first);
      if (f == fresh.end() || f->second.sizes() != kv.second.sizes() || f->second.scalar_type() != kv.second.scalar_type())
        throw std::runtime_error("swap: mismatch at " + kv.first);
    }
    c10::cuda::CUDAStreamGuard guard(s.stream);
    at::NoGradGuard ng;
    for (auto& kv : s.tensors) {
      auto f = fresh.find(kv.first);
      if (f == fresh.end() || f->second.sizes() != kv.second.sizes()) throw std::runtime_error("swap: mismatch at " + kv.first);
      kv.second.copy_(f->second, /*non_blocking=*/false);
    }
    s.stream.synchronize();
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

// in: direct buffer with the bucket layout for (B, T, O); out: direct float buffer, at least B*O floats.
static jint forward_impl(JNIEnv* env, jlong h, jobject in, jint B, jint T, jint O, jint K, jobject out, int abi) {
  try {
    auto& s = *reinterpret_cast<Slot*>(h);
    if (v4(abi) != v4(s.abi) || (v4(abi) && K < 1)) throw std::runtime_error("tensor ABI/grid mismatch");
    void* src = env->GetDirectBufferAddress(in);
    void* dst = env->GetDirectBufferAddress(out);
    if (!src || !dst) throw std::runtime_error("forward: not a direct buffer");
    if (B < 1 || T < 1 || O < 1 || size_t(env->GetDirectBufferCapacity(in)) < layout(B,T,O,K,s.abi).total)
      throw std::runtime_error("forward: invalid input shape/capacity");
    if (env->GetDirectBufferCapacity(out) < int64_t(B) * O) throw std::runtime_error("forward: output buffer too small");
    const auto rows = sparse_rows(src, B, T, O, s.sparse, K, s.abi);
    std::lock_guard<std::mutex> lock(g_gpu);
    c10::InferenceMode im;
    c10::cuda::CUDAStreamGuard guard(s.stream);
    Bucket& b = bucket(s, B, T, O, rows, K);
    if (size_t(env->GetDirectBufferCapacity(in)) < b.lay.total) throw std::runtime_error("forward: input buffer too small");
    copy_input(b, src, rows);
    at::Tensor logits;
    if (b.graph) {
      b.graph->replay();
      logits = b.out;
    } else {
      logits = run(s, b.args);
    }
    b.out_host.copy_(logits, /*non_blocking=*/true);
    b.done.record(s.stream);
    b.done.synchronize();
    std::memcpy(dst, b.out_host.data_ptr(), size_t(B) * O * sizeof(float));
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

// Asynchronous forward: queue copy + replay + copy back on the slot's stream and return; collect() waits (sleeping) and copies
// the logits out. One pending launch per slot; slots run on their own streams, so launches of several slots overlap on the GPU.
static jint launch_impl(JNIEnv* env, jlong h, jobject in, jint B, jint T, jint O, jint K, int abi) {
  try {
    auto& s = *reinterpret_cast<Slot*>(h);
    if (v4(abi) != v4(s.abi) || (v4(abi) && K < 1)) throw std::runtime_error("tensor ABI/grid mismatch");
    void* src = env->GetDirectBufferAddress(in);
    if (!src) throw std::runtime_error("launch: not a direct buffer");
    if (B < 1 || T < 1 || O < 1 || size_t(env->GetDirectBufferCapacity(in)) < layout(B,T,O,K,s.abi).total)
      throw std::runtime_error("launch: invalid input shape/capacity");
    const auto rows = sparse_rows(src, B, T, O, s.sparse, K, s.abi);
    if (s.pending) throw std::runtime_error("launch: previous launch not collected");
    std::lock_guard<std::mutex> lock(g_gpu);
    c10::InferenceMode im;
    c10::cuda::CUDAStreamGuard guard(s.stream);
    Bucket& b = bucket(s, B, T, O, rows, K);
    if (size_t(env->GetDirectBufferCapacity(in)) < b.lay.total) throw std::runtime_error("launch: input buffer too small");
    copy_input(b, src, rows);
    at::Tensor logits;
    if (b.graph) {
      b.graph->replay();
      logits = b.out;
    } else {
      logits = run(s, b.args);
    }
    b.out_host.copy_(logits, /*non_blocking=*/true);
    b.done.record(s.stream);
    b.B = B;
    b.O = O;
    s.pending = &b;
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

JNIEXPORT jint JNICALL Java_com_gempukku_swccgo_bench_NativePolicy_forward(JNIEnv* e, jclass, jlong h, jobject in, jint B, jint T, jint O, jobject out) {
  return forward_impl(e, h, in, B, T, O, 0, out, 3);
}
JNIEXPORT jint JNICALL Java_com_gempukku_swccgo_bench_NativePolicy_launch(JNIEnv* e, jclass, jlong h, jobject in, jint B, jint T, jint O) {
  return launch_impl(e, h, in, B, T, O, 0, 3);
}
JNIEXPORT jint JNICALL Java_com_gempukku_swccgo_bench_NativePolicyV4_forward(JNIEnv* e, jclass, jlong h, jobject in, jint B, jint T, jint O, jint K, jobject out) {
  return forward_impl(e, h, in, B, T, O, K, out, 4);
}
JNIEXPORT jint JNICALL Java_com_gempukku_swccgo_bench_NativePolicyV4_launch(JNIEnv* e, jclass, jlong h, jobject in, jint B, jint T, jint O, jint K) {
  return launch_impl(e, h, in, B, T, O, K, 4);
}
JNIEXPORT jint JNICALL Java_com_gempukku_swccgo_bench_NativePolicyV4_tensorAbi(JNIEnv*, jclass, jlong h) {
  return reinterpret_cast<Slot*>(h)->abi;
}
JNIEXPORT jint JNICALL Java_com_gempukku_swccgo_bench_NativePolicyV4_checkTable(JNIEnv* env, jclass, jlong h, jstring expected) {
  try {
    auto& s = *reinterpret_cast<Slot*>(h);
    std::lock_guard<std::mutex> lock(g_gpu);
    if (!v4(s.abi) || !s.tensors.count("table_sha")) throw std::runtime_error("ABI-4 owned table required");
    auto bytes = s.tensors.at("table_sha").cpu().contiguous();
    const char* hex = "0123456789abcdef";
    std::string actual;
    for (int64_t i = 0; i < bytes.numel(); ++i) {
      uint8_t b = bytes.data_ptr<uint8_t>()[i]; actual += hex[b >> 4]; actual += hex[b & 15];
    }
    if (actual != jstr(env, expected)) throw std::runtime_error("native model/bundle table SHA mismatch");
    return 0;
  } catch (const std::exception& e) { g_error = e.what(); return -1; }
}

JNIEXPORT jint JNICALL Java_com_gempukku_swccgo_bench_NativePolicy_collect(JNIEnv* env, jclass, jlong h, jobject out) {
  try {
    auto& s = *reinterpret_cast<Slot*>(h);
    void* dst = env->GetDirectBufferAddress(out);
    if (!dst) throw std::runtime_error("collect: not a direct buffer");
    Bucket* b = s.pending;
    if (!b) throw std::runtime_error("collect: nothing launched");
    if (env->GetDirectBufferCapacity(out) < int64_t(b->B) * b->O) throw std::runtime_error("collect: output buffer too small");
    b->done.synchronize();
    std::memcpy(dst, b->out_host.data_ptr(), size_t(b->B) * b->O * sizeof(float));
    s.pending = nullptr;
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

JNIEXPORT void JNICALL Java_com_gempukku_swccgo_bench_NativePolicy_free(JNIEnv*, jclass, jlong h) {
  std::lock_guard<std::mutex> lock(g_gpu);
  delete reinterpret_cast<Slot*>(h);
}

}  // extern "C"

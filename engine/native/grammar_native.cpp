#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <mlx/allocator.h>
#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/backend/cpu/encoder.h>
#include <mlx/primitives.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <typeinfo>
#include <unordered_map>

namespace nb = nanobind;
namespace mx = mlx::core;

struct Context {
    nb::object callback;
    std::mutex mutex;
    std::condition_variable cv;
    uint64_t submitted = 0, completed = 0;
    std::unique_ptr<nb::python_error> error;
    std::string native_error;

    explicit Context(nb::object fn) : callback(std::move(fn)) {}
    ~Context() {
        nb::gil_scoped_acquire gil;
        error.reset();
        callback.reset();
    }
};

static std::mutex registry_mutex;
static std::unordered_map<uint64_t, std::shared_ptr<Context>> registry;
static std::atomic<uint64_t> next_id{1};

static std::shared_ptr<Context> context(uint64_t id) {
    std::lock_guard lock(registry_mutex);
    return registry.at(id);
}

class Mask : public mx::Primitive {
public:
    explicit Mask(mx::Stream stream) : mx::Primitive(stream) {}

    const char* name() const override { return "XGrammarMask"; }

    void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("XGrammarMask requires a CPU stream");
    }

    void eval_cpu(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outputs) override {
        auto ctx = context(*inputs[2].data<uint64_t>());
        auto& out = outputs[0];
        out.set_data(mx::allocator::malloc(out.nbytes()));
        uint64_t ticket;
        {
            std::lock_guard lock(ctx->mutex);
            ticket = ++ctx->submitted;
        }
        auto token = inputs[0], active = inputs[1], mask = out;
        mx::cpu::get_command_encoder(stream()).dispatch(
            [ctx, token, active, mask, ticket]() mutable {
                nb::gil_scoped_acquire gil;
                auto* dest = mask.data<uint32_t>();
                std::memset(dest, 0xff, mask.nbytes());
                if (!ctx->error && ctx->native_error.empty()) {
                    try {
                        auto value = ctx->callback(*token.data<int32_t>(), *active.data<bool>());
                        auto data = nb::cast<nb::ndarray<nb::numpy, const uint32_t, nb::c_contig>>(value);
                        if (data.size() + 1 != mask.size()) {
                            throw std::runtime_error("Grammar mask size mismatch");
                        }
                        std::memcpy(dest, data.data(), data.size() * sizeof(uint32_t));
                    } catch (nb::python_error& error) {
                        // Preserve the traceback across the executor boundary for drain().
                        ctx->error = std::make_unique<nb::python_error>(std::move(error));
                    } catch (const std::bad_cast&) {
                        ctx->native_error = "Grammar callback must return a contiguous uint32 mask";
                    } catch (const std::runtime_error& error) {
                        ctx->native_error = error.what();
                    }
                }
                // A failed callback disables sampling in the rest of this packet.
                dest[mask.size() - 1] = (!ctx->error && ctx->native_error.empty()) ? 1u : 0u;
                {
                    std::lock_guard lock(ctx->mutex);
                    ctx->completed = ticket;
                }
                ctx->cv.notify_all();
            });
    }
};

static uint64_t create(nb::object callback) {
    auto id = next_id.fetch_add(1);
    auto ctx = std::make_shared<Context>(std::move(callback));
    std::lock_guard lock(registry_mutex);
    registry.emplace(id, std::move(ctx));
    return id;
}

static void wait_context(const std::shared_ptr<Context>& ctx) {
    nb::gil_scoped_release gil;
    std::unique_lock lock(ctx->mutex);
    ctx->cv.wait(lock, [&] { return ctx->submitted == ctx->completed; });
}

static void drain(uint64_t id) {
    auto ctx = context(id);
    wait_context(ctx);
    if (ctx->error) {
        ctx->error->restore();
        throw nb::python_error();
    }
    if (!ctx->native_error.empty()) {
        throw std::runtime_error(ctx->native_error);
    }
}

static void close_context(uint64_t id) {
    auto ctx = context(id);
    wait_context(ctx);
    std::lock_guard lock(registry_mutex);
    registry.erase(id);
}

static mx::array mask(const mx::array& token, const mx::array& active,
                      const mx::array& handle, int words) {
    return mx::array({words + 1}, mx::uint32,
                     std::make_shared<Mask>(mx::default_stream(mx::Device::cpu)),
                     {token, active, handle});
}

class Handoff : public mx::Primitive {
public:
    explicit Handoff(mx::Stream stream) : mx::Primitive(stream) {}
    const char* name() const override { return "XGrammarHandoff"; }
    void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
        throw std::runtime_error("XGrammarHandoff requires a GPU stream");
    }
    void eval_gpu(const std::vector<mx::array>& inputs,
                  std::vector<mx::array>& outputs) override {
        // Unified-memory alias: the evaluator has already ordered the CPU fence.
        outputs[0].copy_shared_buffer(inputs[0]);
    }
};

// The marker reads the forward's output, so MLX inserts a Metal buffer barrier
// before dispatching it. Only then may the CPU mask's GPU fence be encoded.
// An alias alone would not insert that GPU buffer barrier.
static mx::array handoff(const mx::array& mask, const mx::array& completed) {
    static const auto barrier = mx::fast::metal_kernel(
        "xgrammar_forward_completed", {"completed"}, {"marker"},
        "marker[0] = uint(completed[0] != 0);");
    auto stream = mx::default_stream(mx::Device::gpu);
    auto marker = barrier({completed}, {{1}}, {mx::uint32}, {1, 1, 1}, {1, 1, 1}, {},
                          std::nullopt, false, stream)[0];
    return mx::array(mask.shape(), mask.dtype(), std::make_shared<Handoff>(stream),
                     {mask, marker});
}

NB_MODULE(_grammar_native, m) {
    nb::module_::import_("mlx.core");
    m.def("create", &create);
    m.def("drain", &drain);
    m.def("close", &close_context);
    m.def("mask", &mask);
    m.def("handoff", &handoff);
}

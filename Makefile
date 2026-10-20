CXX := g++
CXXFLAGS := -std=c++20 -O2 -g -Wall -Wextra -Werror -fPIC -fvisibility=hidden -Iinclude
ORT_INC := third_party/onnxruntime/core/session
BUILD := build
VENV := /tmp/accel-venv
PY := $(VENV)/bin/python

RUNTIME_OBJS := $(BUILD)/device_buffer.o $(BUILD)/sim_device.o $(BUILD)/runtime.o
KERNEL_OBJS := $(BUILD)/sim_embed.o $(BUILD)/sim_rmsnorm.o $(BUILD)/sim_rope.o $(BUILD)/sim_cache.o $(BUILD)/sim_attend.o $(BUILD)/sim_silu.o $(BUILD)/sim_add.o $(BUILD)/sim_sample.o

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
SIM_LIB := $(BUILD)/libaccel_sim.dylib
SIM_SHARED := -shared
else
SIM_LIB := $(BUILD)/libaccel_sim.so
SIM_SHARED := -shared -Wl,-z,defs
endif

.PHONY: all test venv bench clean simlib sim-test

all: $(BUILD)/libaccel_ep.so $(BUILD)/test_runtime

simlib: $(SIM_LIB)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/device_buffer.o: src/device_buffer.cc include/accel/device.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ src/device_buffer.cc

$(BUILD)/sim_device.o: src/sim_device.cc include/accel/device.h include/accel/runtime.h include/accel/sim_add.h include/accel/sim_attend.h include/accel/sim_cache.h include/accel/sim_embed.h include/accel/sim_rmsnorm.h include/accel/sim_rope.h include/accel/sim_sample.h include/accel/sim_silu.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ src/sim_device.cc

$(KERNEL_OBJS): $(BUILD)/sim_%.o: src/sim_%.cc include/accel/sim_%.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD)/runtime.o: src/runtime.cc include/accel/runtime.h include/accel/device.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ src/runtime.cc

$(BUILD)/ep.o: src/ep.cc include/accel/runtime.h include/accel/device.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -isystem $(ORT_INC) -c -o $@ src/ep.cc

$(BUILD)/libaccel_ep.so: $(RUNTIME_OBJS) $(KERNEL_OBJS) $(BUILD)/ep.o
	$(CXX) -shared -Wl,-z,defs -o $@ $^

$(BUILD)/sim_api.o: src/sim_api.cc include/accel/sim_api.h include/accel/device.h include/accel/runtime.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ src/sim_api.cc

$(BUILD)/llama_model.o: src/llama_model.cc include/accel/sim_api.h include/accel/device.h include/accel/bpe.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ src/llama_model.cc

$(BUILD)/bpe.o: src/bpe.cc include/accel/bpe.h src/unicode_cats.inc | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ src/bpe.cc

$(SIM_LIB): $(RUNTIME_OBJS) $(KERNEL_OBJS) $(BUILD)/sim_api.o $(BUILD)/llama_model.o $(BUILD)/bpe.o
	$(CXX) $(SIM_SHARED) -o $@ $^

$(BUILD)/test_runtime: tests/test_runtime.cc tests/recording_device.cc tests/recording_device.h $(RUNTIME_OBJS) $(KERNEL_OBJS)
	$(CXX) -std=c++20 -O2 -g -Wall -Wextra -Werror -fPIC -Iinclude -Itests -o $@ tests/test_runtime.cc tests/recording_device.cc $(RUNTIME_OBJS) $(KERNEL_OBJS)

$(BUILD)/test_sim_kernels: tests/test_sim_kernels.cc $(KERNEL_OBJS)
	$(CXX) -std=c++20 -O2 -g -Wall -Wextra -Werror -Iinclude -o $@ tests/test_sim_kernels.cc $(KERNEL_OBJS)

venv:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --disable-pip-version-check 'onnxruntime==1.31.0' 'onnx==1.21.0'

test: all
	./$(BUILD)/test_runtime
	$(PY) tests/test_exports.py
	$(PY) tests/test_ep.py
	$(PY) tests/test_ep_reject.py
	$(PY) tests/test_bench.py
	$(PY) tests/test_stack_matmul.py

sim-test: $(BUILD)/test_sim_kernels $(BUILD)/test_runtime $(SIM_LIB)
	./$(BUILD)/test_sim_kernels
	./$(BUILD)/test_runtime
	@test -x .venv/bin/python || { echo "missing .venv. Run: python3 -m venv .venv && .venv/bin/pip install numpy tokenizers"; exit 1; }
	PYTHONPATH=. .venv/bin/python tests/test_serve.py

bench:
	$(PY) bench/mlp_bench.py

clean:
	rm -rf $(BUILD)

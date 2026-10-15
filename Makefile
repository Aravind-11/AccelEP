CXX := g++
CXXFLAGS := -std=c++20 -O2 -g -Wall -Wextra -Werror -fPIC -fvisibility=hidden -Iinclude
ORT_INC := third_party/onnxruntime/core/session
BUILD := build
VENV := /tmp/accel-venv
PY := $(VENV)/bin/python

RUNTIME_OBJS := $(BUILD)/device_buffer.o $(BUILD)/sim_device.o $(BUILD)/runtime.o

.PHONY: all test venv clean

all: $(BUILD)/libaccel_ep.so $(BUILD)/test_runtime

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/device_buffer.o: src/device_buffer.cc include/accel/device.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ src/device_buffer.cc

$(BUILD)/sim_device.o: src/sim_device.cc include/accel/device.h include/accel/runtime.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ src/sim_device.cc

$(BUILD)/runtime.o: src/runtime.cc include/accel/runtime.h include/accel/device.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -c -o $@ src/runtime.cc

$(BUILD)/ep.o: src/ep.cc include/accel/runtime.h include/accel/device.h | $(BUILD)
	$(CXX) $(CXXFLAGS) -isystem $(ORT_INC) -c -o $@ src/ep.cc

$(BUILD)/libaccel_ep.so: $(RUNTIME_OBJS) $(BUILD)/ep.o
	$(CXX) -shared -Wl,-z,defs -o $@ $^

$(BUILD)/test_runtime: tests/test_runtime.cc $(RUNTIME_OBJS)
	$(CXX) -std=c++20 -O2 -g -Wall -Wextra -Werror -fPIC -Iinclude -o $@ tests/test_runtime.cc $(RUNTIME_OBJS)

venv:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --disable-pip-version-check 'onnxruntime==1.31.0' 'onnx==1.21.0'

test: all
	./$(BUILD)/test_runtime
	$(PY) tests/test_ep.py

clean:
	rm -rf $(BUILD)

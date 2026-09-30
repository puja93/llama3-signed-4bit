CXX = clang++
CXXFLAGS = -O3 -std=c++20 -mcpu=apple-m1

TARGET = build/chat
DYLIB = build/libllama_engine.dylib

all: $(DYLIB) $(TARGET)

$(DYLIB): src/llama_engine.cpp src/llama_engine.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -dynamiclib -fPIC $< -o $@

$(TARGET): src/main.cpp $(DYLIB)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -Isrc $< $(DYLIB) -Wl,-rpath,@loader_path -o $@

test: $(DYLIB)
	/opt/anaconda3/bin/python3 -m unittest discover -s tests -p "test_*.py" -v

clean:
	rm -rf build

.PHONY: all test clean




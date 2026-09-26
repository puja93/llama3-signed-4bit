CXX = clang++
CXXFLAGS = -O3 -std=c++20 -mcpu=apple-m1

TARGET = build/chat

all: $(TARGET)

$(TARGET): src/main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf build

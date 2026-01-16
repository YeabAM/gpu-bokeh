# Compiler
NVCC = nvcc

# Flags
CUDA_FLAGS = -O3 -arch=sm_80
INCLUDE_FLAGS = -I./include

# Source files
SOURCES = src/main.cu src/kernel.cu src/support.cu

# Output
TARGET = bokeh

# Build
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDE_FLAGS) $^ -o $@

# Clean
clean:
	rm -f $(TARGET)

# Run (default settings)
run: $(TARGET)
	./$(TARGET) --input data --output output --blur-radius 15

# Help
help:
	@echo "Usage:"
	@echo "  make          - Build the project"
	@echo "  make clean    - Remove executable"
	@echo "  make run      - Build and run with defaults"
	@echo "  make help     - Show this message"
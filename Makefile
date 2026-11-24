CC ?= gcc

BUILD ?= release
BUILD_DIR = build/$(BUILD)

COMMON_FLAGS = -std=c99 -Wall -Wextra -Wpedantic -Wno-unused -Wconversion -Wsign-conversion -Wshadow -Wformat=2 -Wnull-dereference -Wdouble-promotion -Wstrict-prototypes -Wmissing-prototypes -Wold-style-definition -Wundef -Wcast-qual -Wwrite-strings

SOURCES = $(wildcard src/*.c)
OBJECTS = $(patsubst src/%.c,build/$(BUILD)/%.o,$(SOURCES))

TARGET = build/$(BUILD)/main

DEPFLAGS = -MMD -MP
DEPS = $(OBJECTS:.o=.d)

ifeq ($(BUILD),debug)
	OPT_FLAGS = -O0 -g3 -fno-omit-frame-pointer
	SAN_FLAGS = -fsanitize=address,undefined -fno-sanitize=leak -fno-sanitize-recover=all

else ifeq ($(BUILD),release)
	OPT_FLAGS = -O3
	SAN_FLAGS = 

else ifeq ($(BUILD),relwithdebinfo)
	OPT_FLAGS = -O2 -g
	SAN_FLAGS = 

else ifeq ($(BUILD),sanitize)
	OPT_FLAGS = -O0 -g3 -fno-omit-frame-pointer
	SAN_FLAGS = -fsanitize=address,undefined -fno-sanitize=leak -fno-sanitize-recover=all

else ifeq ($(BUILD),valgrind)
    OPT_FLAGS = -O0 -g3 -fno-omit-frame-pointer
    SAN_FLAGS =
	
else
	$(error Unknown build config: $(BUILD))
endif
	
CFLAGS  = $(COMMON_FLAGS) $(OPT_FLAGS) $(SAN_FLAGS)
LDFLAGS = -flto $(SAN_FLAGS)
LDLIBS  = -lm

.PHONY: all clean run

$(BUILD_DIR)/%.o: src/%.c
	@mkdir -p $(BUILD_DIR)
	@$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	@$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

all: $(TARGET)

run: $(TARGET)
	@./$(TARGET)

clean:
	@rm -rf build

-include $(DEPS)
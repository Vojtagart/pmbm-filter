CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wfatal-errors -pedantic -Wextra -g -MMD -MP
CXXFLAGS += -O3 -march=native
VALGRIND_FLAGS := --leak-check=full --track-origins=yes

# Path to the Eigen header files
EIGEN_DIR = /usr/local/include/eigen5
CXXFLAGS += -I$(EIGEN_DIR)

# Turns off extra checks inside the program
# CXXFLAGS += -D NDEBUG

SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)
DEPS := $(OBJS:.o=.d)

BIN := pmbm

.PHONY: all clean run doc memcheck

all: $(BIN)

$(BIN): $(OBJS)
	@$(CXX) $(OBJS) -o $(BIN)

%.o: %.cpp
	@echo "Compiling $<.."
	@$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(BIN)
	@echo "Running.."
	@./$(BIN)

memcheck: $(BIN)
	@echo "Running with Valgrind.."
	valgrind $(VALGRIND_FLAGS) ./$(BIN)

doc:
	@echo "Generating documentation.."
	@rm -rf ./doc
	@doxygen Doxyfile > /dev/null

clean:
	@echo "Cleaning..."
	@rm -f $(BIN) $(OBJS) $(DEPS)
	@rm -rf ./doc

-include $(DEPS)
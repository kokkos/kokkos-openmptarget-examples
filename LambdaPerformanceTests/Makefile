KOKKOS_PATH = ${KOKKOS}/rgayatri-kokkos
SRC = $(wildcard *.cpp)
vpath %.cpp $(sort $(dir $(SRC)))
HEADERS = axpby.hpp dot.hpp

default: build
	echo "Start Build"

CXXFLAGS = -O3
ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
LINK = ${CXX}
LDFLAGS = 
EXE = test.cuda
KOKKOS_DEVICES = "Cuda"
KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS += "enable_lambda"
else
CXX = clang++
LINK = ${CXX}
EXE = test.ompt
KOKKOS_DEVICES = "OpenMPTarget"
KOKKOS_ARCH = "Volta70"
KOKKOS_CXX_STANDARD=c++17
CXXFLAGS += -Wno-unknown-cuda-version -Wno-undefined-internal -Wno-pass-failed -Wno-openmp-mapping -Wno-extra-tokens
CXXFLAGS += -fopenmp-cuda-mode
LINK = ${CXX}
LDFLAGS =-Wno-unknown-cuda-version
CXXFLAGS += $(LLVM_CFLAGS)
LDFLAGS += $(LLVM_CFLAGS)
endif

DEPFLAGS = -M

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

test: $(EXE)
	./$(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LDFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host *.ompt

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HEADERS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)

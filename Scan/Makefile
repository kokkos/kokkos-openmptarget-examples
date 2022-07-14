#KOKKOS_PATH = ${KOKKOS}/lammps-kokkos
KOKKOS_PATH = ${HOME}/Kokkos/rgayatri-kokkos
SRC = $(wildcard *.cpp)
vpath %.cpp $(sort $(dir $(SRC)))
HEADER=scan.hpp

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS = -O3
LINK = ${CXX}
LDFLAGS = 
EXE = test.cuda
KOKKOS_DEVICES = "Cuda"
KOKKOS_ARCH = "Volta70"
KOKKOS_CUDA_OPTIONS += "enable_lambda"
else
CXX = clang++
CXXFLAGS = -O3
LINK = ${CXX}
EXE = test.ompt
KOKKOS_DEVICES = "OpenMPTarget"
KOKKOS_ARCH = "Volta70"
KOKKOS_CXX_STANDARD=c++17
#CXXFLAGS += --gcc-toolchain=/global/cfs/cdirs/m1759/rgayatri/software/GCC/gcc-10.2/install
#CXXFLAGS += -Wno-unknown-cuda-version -Wno-undefined-internal -Wno-pass-failed -Wno-openmp-mapping -Wno-extra-tokens
#CXXFLAGS += -fopenmp-cuda-mode
LINK = ${CXX}
LDFLAGS =-Wno-unknown-cuda-version
#CXXFLAGS += $(LLVM_CFLAGS)
#LDFLAGS += $(LLVM_CFLAGS)
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
	rm -f *.o *.ompt *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HEADER)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)

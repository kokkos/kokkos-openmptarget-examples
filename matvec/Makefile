KOKKOS_PATH = ${HOME}/Kokkos/kokkos
SRC = $(wildcard *.cpp)
KOKKOS_CUDA_OPTIONS=enable_lambda
KOKKOS_ARCH = Volta70

HEADER = matvec.hpp

default: build
	echo "Start Build"

EXE = test.ompt
KOKKOS_CXX_STANDARD=c++17

CXXFLAGS = -O3 -g

# To avoid warnings
ifeq ($(DEVICE),cuda)
KOKKOS_DEVICES = "Cuda"
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
else
KOKKOS_DEVICES=OpenMPTarget
CXX = clang++
CXXFLAGS += -Wno-unknown-cuda-version -Wno-undefined-internal -Wno-pass-failed -Wno-openmp-mapping -Wno-extra-tokens
LINKFLAGS =-Wno-unknown-cuda-version
endif

LINK = ${CXX}

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host *.ompt

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HEADER)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

KOKKOS_PATH = ${HOME}/Kokkos/rgayatri-kokkos
SRC = $(wildcard *.cpp)
vpath %.cpp $(sort $(dir $(SRC)))

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
LINK = ${CXX}
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

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) 
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)

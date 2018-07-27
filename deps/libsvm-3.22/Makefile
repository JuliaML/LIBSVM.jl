CXX ?= g++
CFLAGS_normal = -Wall -Wconversion -O3 -fPIC -DENABLEOPENMP -fopenmp
CFLAGS_fallback = -Wall -Wconversion -O3 -DDISABLEOPENMP -fPIC
SHVER = 2
OS = $(shell uname)

all: svm-train svm-predict svm-scale

lib:
	$(MAKE) lib_normal || $(MAKE) lib_fallback

lib_normal: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} -DENABLEOPENMP -fopenmp svm.o -o libsvm.so.$(SHVER)

lib_fallback: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} -DDISABLEOPENMP svm.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.c svm.o
	$(CXX) $(CFLAGS) svm-predict.c svm.o -o svm-predict -lm

svm-train: svm-train.c svm.o
	$(CXX) $(CFLAGS) svm-train.c svm.o -o svm-train -lm

svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale

svm.o:
	$(MAKE) svm.o_normal || $(MAKE) svm.o_fallback

svm.o_normal: svm.cpp svm.h
	$(CXX) $(CFLAGS_normal) -c svm.cpp

svm.o_fallback: svm.cpp svm.h
	$(CXX) $(CFLAGS_fallblack) -c svm.cpp

clean:
	rm -f *~ svm.o svm-train svm-predict svm-scale libsvm.so.$(SHVER)

TARGET := libSubsense.dll
LDFLAGS := -shared

#/cygdrive/c/cygwin64/usr/include/
#/cygdrive/c/cygwin64/usr/x86_64-w64-mingw32/sys-root/mingw/lib
OPENCV_INC := -I"C:\cygwin64\usr\include"
OPENCV_LIB := -L"C:\cygwin64\usr\lib" -lopencv_core -lopencv_video -lopencv_imgproc -lopencv_features2d

SRCS_FILES := $(wildcard subsense_cpp/*.cpp)

all:
	g++ -std=c++11 $(SRCS_FILES) $(LDFLAGS) $(OPENCV_INC) $(OPENCV_LIB) -o $(TARGET)
	
clean :
	@rm -f *.o $(TARGET)
	
.PHONY : all clean
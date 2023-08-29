all:
	nvcc  -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgcodecs main.cpp kernel.cu -o executable
	# './executable'

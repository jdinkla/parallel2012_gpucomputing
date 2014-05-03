/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#include "CExtent.h"
#include "CTimed.h"

#include <iostream>
using namespace std;

extern void smoothing1(const CExtent& extent);
extern void smoothing2(const CExtent& extent);
extern void smoothing3(const CExtent& extent);
extern void smoothCPU(const CExtent& extentHost, const int windowSize);
extern void smoothing3cu(const CExtent& extent, const int windowSize);

extern void smoothing3b(const CExtent& extent);
extern void smoothing3d(const CExtent& extent);
extern void smoothing3d_timing(const CExtent& extent);
extern void smoothing3d_swap(const CExtent& extentHost, const CExtent& extentDevice, const CExtent& extentPartition, const int windowSize);
extern void smoothing3d_swap2(const CExtent& extentHost, const CExtent& extentDevice, const CExtent& extentPartition, const int windowSize);
extern void smoothing3d_stream(const CExtent& extentHost, const CExtent& extentDevice, const CExtent& extentPartition, const int windowSize);
extern void smoothing3d_stream2(const CExtent& extentHost, const CExtent& extentDevice, const CExtent& extentPartition, const int windowSize);
extern void smoothing3d_stream3(const CExtent& extentHost, const CExtent& extentDevice, const CExtent& extentPartition, const int windowSize);
extern void smoothing3d_stream4(const CExtent& extentHost, const CExtent& extentDevice, const CExtent& extentPartition, const int windowSize);
extern void smoothing3d_stream5(
					  const CExtent& extentHost, 
					  const CExtent& extentDevice, 
					  const CExtent& extentPartition, 
					  const int windowSize);

bool compareWithCPU = false;

// 2D
void example_1() {
	cout << "smoothing1" << endl;
	compareWithCPU = true;
	const int h_size = 1024;
	CExtent extentHost(h_size, h_size, 1);
	smoothing1(extentHost);
}

void example_2() {
	cout << "smoothing2" << endl;
	compareWithCPU = true;
	const int h_size = 1024;
	CExtent extentHost(h_size, h_size, 1);
	smoothing2(extentHost);
}

void example_3() {
	cout << "smoothing3" << endl;
	compareWithCPU = true;
	const int h_size = 1024;
	CExtent extentHost(h_size, h_size, 1);
	smoothing3(extentHost);
}

// 3D Naiv
void example3DNaive() {
	cout << "example3DNaive" << endl;
	compareWithCPU = true;
	const int h_size = 512;
	const int windowSize = 1;
	CExtent extentHost(h_size, h_size, h_size);
	//smoothing3d(extentHost);
	smoothing3d_timing(extentHost);
}

// 3D mit Swapping
void example3DSwapping() {
	cout << "example3DSwapping" << endl;
	compareWithCPU = false;
	const int h_size = 768;
	const int d_size = 256;
	//const int h_size = 8;
	//const int d_size = 4;
	const int windowSize = 1;
	CExtent extentHost(h_size, h_size, h_size);
	CExtent extentPartition(h_size, h_size, d_size);
	CExtent extentDevice(h_size, h_size, d_size + 2 * windowSize);  // Padding
	smoothing3d_swap2(extentHost, extentDevice, extentPartition, windowSize);
}

// 3D + Streaming + Swapping
void example3DStreaming() {
	cout << "example3DStreaming" << endl;
	compareWithCPU = false; //true;
	const int h_size = 512; // 1024, 1152, 1280;
	const int d_size = 128; // 256, 
	const int windowSize = 1;
	CExtent extentHost(h_size, h_size, h_size);
	CExtent extentPartition(h_size, h_size, d_size);
	CExtent extentDevice(h_size, h_size, d_size + 2 * windowSize);  // Padding
	//smoothing3d_stream(extentHost, extentDevice, extentPartition, windowSize);
	//smoothing3d_stream2(extentHost, extentDevice, extentPartition, windowSize);
	//smoothing3d_stream3(extentHost, extentDevice, extentPartition, windowSize);
	//smoothing3d_stream4(extentHost, extentDevice, extentPartition, windowSize);
	smoothing3d_stream5(extentHost, extentDevice, extentPartition, windowSize);
}


void performance3D_1() {
	cout << "performance3D_1" << endl;
	compareWithCPU = true;
	int sizes[] = { 8, 16, 32, 64, 128, 256, 384, 512, 768 };
	for (int i=0; i<9; i++) {
		const int h_size = sizes[i];
		CExtent extent(h_size, h_size, h_size);
		const int windowSize = 1;
		smoothing3cu(extent, windowSize);
	}
}

void performance3D_2() {
	cout << "performance3D_2" << endl;
	compareWithCPU = true;
	const int windowSize = 1;
	const int sizes[] = { 768, 1024, 1152, 1280 };
	for (int i=0; i<9; i++) {
		const int h_size = sizes[i];
		const int d_size = 128;
		CExtent extent(h_size, h_size, h_size);
		CExtent extentHost(h_size, h_size, h_size);
		CExtent extentPartition(h_size, h_size, d_size);
		CExtent extentDevice(h_size, h_size, d_size + 2 * windowSize);  // Padding
		smoothing3d_stream5(extentHost, extentDevice, extentPartition, windowSize);
	}
}

void performance3D_CPU() {
	cout << "performance3D_CPU" << endl;
	// int sizes[] = { 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1152, 1280 };
//	int sizes[] = { 768, 1024, 1152, 1280 };
	int sizes[] = { 1152 };
	for (int i=0; i<1; i++) {
		const int h_size = sizes[i];
		CExtent extent(h_size, h_size, h_size);
		const int windowSize = 1;
		smoothCPU(extent, windowSize);
	}
}



void main(int argc, char** argv) {

	CTimed timer;
	for (int i=1; i<argc; i++) {
		const int n = atoi(argv[i]);
		cout << "n=" << n << endl;

		timer.start();
		try	{
			switch(n) {
				case 1:
					example_1();
					break;
				case 2:
					example_2();
					break;
				case 3:
					example_3();
					break;
				case 4:
					example3DNaive();
					break;
				case 5:
					example3DSwapping();
					break;
				case 6:
					example3DStreaming();
					break;
				case 7:
					performance3D_1();
					break;
				case 8:
					performance3D_2();
					break;
				case 9:
					performance3D_CPU();
					break;
				default:
					cerr << "ERROR: unknown algorithm id " << n << endl;
					exit(1);
			}
		} catch (int e) {
			cerr << "EXCEPTION: " << e << endl;
		}
		timer.stop();
		cout << "Overall: " << timer.getDuration() << endl;
	}
	return;
}

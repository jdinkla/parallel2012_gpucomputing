/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifdef _MSC_VER
#include <windows.h>
#include <numeric>
#include <sstream>
#include <string>
#include <ctime>
#include <iomanip>
#endif

#include "OSUtilities.h"
#include "Defs.h"

using namespace std;

// see http://en.allexperts.com/q/C-1040/time-milliseconds-Windows.htm
uint64 GetCurrentSystemTime()
{
#ifdef _MSC_VER
	FILETIME now;
	GetSystemTimeAsFileTime(&now);

	ULARGE_INTEGER uli;
	uli.LowPart = now.dwLowDateTime;
	uli.HighPart = now.dwHighDateTime;

	return (uint64) (uli.QuadPart/10000);
#elif __APPLE__
	return 1;
#else
	return 0;
#endif

}


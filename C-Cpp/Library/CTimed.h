/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CTIMED_H
#define CTIMED_H

#include "Defs.h"
#include "OSUtilities.h"

class CTimed 
{

private:

	uint64 startTime;
	uint64 endTime;

public:
	
	CTimed() : startTime(0), endTime(0) {
	}

	void start() {
		startTime = GetCurrentSystemTime();
	}

	void stop() {
		endTime = GetCurrentSystemTime();
	}

	uint64 getDuration() {
		return endTime - startTime;
	}

};

#endif

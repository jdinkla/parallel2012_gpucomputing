/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CVERSION_H
#define CVERSION_H

class CVersion {

	int version;	

public:

	CVersion() 
		: version(-1) {
	}

	void incVersion() {
		version++;
	}

	int getVersion() const {
		return version;
	}

	void resetVersion() {
		version = -1;
	}
};

#endif

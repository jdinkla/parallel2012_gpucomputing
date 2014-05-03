/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef CBUFFERITERATOR_H
#define CBUFFERITERATOR_H

#include <vector_types.h>
#include <assert.h>
#include <cutil_math.h>
#include "Defs.h"
#include "EAxis.h"
#include "EPosition.h"
#include "CExtent.h"

/*
class CBufferIterator {
public:
	CExtent& hostExtent; CExtent& deviceExtent;
	int3 padBefore;	int3 padAfter;					

	CBufferIterator(CExtent& _h,  CExtent& _d);
	int3 getCurrentStart();
	int3 getCurrentEnd()
	int3 getCurrentSize();
	int3 getCurrentStartPadded();
	int3 getCurrentEndPadded();
	int3 getCurrentSizePadded();
	bool isFirstBlock();
	bool isLastBlock();
	bool next();
*/


class CBufferIterator {

public:

	const CExtent& hostExtent;
	const CExtent& deviceExtent;

	int3 host;
	int3 device;

	bool hasNext;					// Is there a next block?
	int3 currentPos;				// The current start position.
	int3 nextPos;					// The next start position.
	int3 padBefore;					// The padding of the block.
	int3 padAfter;					// The padding of the block.
	uint nexts;						// Number of calls to next()
	
	EAxis axes[3];					// The order of the axes.

	CBufferIterator(const CExtent& _host, const CExtent& _device) 
		: hostExtent(_host)
		, deviceExtent(_device)
	{
		host = make_int3(_host.width, _host.height, _host.depth);
		device = make_int3(_device.width, _device.height, _device.depth);
		axes[0] = X;
		axes[1] = Y;
		axes[2] = Z;
		currentPos = make_int3(0, 0, 0);
		nextPos = make_int3(0, 0, 0);
		padBefore = make_int3(0, 0, 0);
		padAfter = make_int3(0, 0, 0);
		reset();
	}

	void reset() {
		nextPos = make_int3(0, 0, 0);
		hasNext = true;
		nexts = 0;
	}

	CBufferIterator copy() {
		CBufferIterator it(hostExtent, deviceExtent);
		it.hasNext = hasNext;
		it.currentPos = currentPos;
		it.nextPos = nextPos;
		it.padBefore = padBefore;
		it.padAfter = padAfter;
		it.axes[0] = axes[0];
		it.axes[1] = axes[1];
		it.axes[2] = axes[2];
		it.nexts = nexts;
		return it;
	}

	int3 getCurrentStart() const {
		return currentPos;
	}

	int3 getCurrentEnd() const {
		return min(currentPos + device - make_int3(1, 1, 1), host - make_int3(1, 1, 1));
	}

	int3 getCurrentSize() const {
		return getCurrentEnd() - getCurrentStart() + make_int3(1, 1, 1);
	}

	int3 getCurrentStartPadded() const {
		int3 start = getCurrentStart();
		if (!isFirstBlock()) {
			start = start - padBefore;
		}
		return start;
	}

	int3 getCurrentEndPadded() const {
		int3 end = getCurrentEnd();
		if (!isLastBlock()) {
			end = end + padAfter;
		}
		return end;
	}

	int3 getCurrentSizePadded() const {
		return getCurrentEndPadded() - getCurrentStartPadded() + make_int3(1, 1, 1);
	}

	bool isFirstBlock() const {
		return (nexts == 1);
	}

	bool isLastBlock() const {
		return (hasNext == false);
	}

	EPosition getPosition() const {
		if (isFirstBlock()) {
			return FIRST;
		} else if (isLastBlock()) {
			return LAST;
		} else {
			return MIDDLE;
		}
	}

	bool next() {
		return next(device);
	}

	bool next(const int3& blockSize)
	{
		device = blockSize;
		nexts++;
		if (hasNext) {
			currentPos = nextPos;		// One step further and calculate the new nextPos
			EAxis axis = axes[0];
			int distance = get<int>(axis, host) - get<int>(axis, currentPos);
			if (distance > get<int>(axis, device)) {
				set(axis, nextPos, get<int>(axis, currentPos) + get<int>(axis, device));
			} else {
				set<int>(axis, nextPos, 0);
				axis = axes[1];
				distance = get<int>(axis, host) - get<int>(axis, currentPos);
				if (distance > get<int>(axis, device)) {
					set(axis, nextPos, get<int>(axis, currentPos) + get<int>(axis, device));
				} else {
					set<int>(axis, nextPos, 0);
					axis = axes[2];
					distance = get<int>(axis, host) - get<int>(axis, currentPos);
					if (distance > get<int>(axis, device)) {
						set(axis, nextPos, get<int>(axis, currentPos) + get<int>(axis, device));
					} else {
						hasNext = false;
					}
				}
			}
			return true;
		} 
		else {
			return false;
		}
	}

	bool next(const int steps) {
		bool hasNext = true;
		int i = 0;
		while (hasNext && i<steps) {
			hasNext = next();
			i++;
		}
		return hasNext;
	}

	void setAxes(const EAxis x, const EAxis y, const EAxis z) {
		assert(x != y && x != z && y != z);
		axes[0] = x;
		axes[1] = y;
		axes[2] = z;
	}


};

#endif

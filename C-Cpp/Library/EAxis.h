/*
 * Copyright (c) 2012 by Jörn Dinkla, www.dinkla.com, All rights reserved.
 */

#ifndef EAXIS_H
#define EAXIS_H

enum EAxis 
{
	X = 1, 
	Y = 2,
	Z = 4
};

template <class V, class T>
V get(const EAxis& axis, const T& triple)
{
	switch(axis) 
	{
	case X:
		return triple.x;
	case Y:
		return triple.y;
	case Z:
		return triple.z;
	}
	return (V)0;
}

template <class V, class T>
void set(const EAxis& axis, T& triple, const V value)
{
	switch(axis) 
	{
	case X:
		triple.x = value;
		break;
	case Y:
		triple.y = value;
		break;
	case Z:
		triple.z = value;
		break;
	}
}


#endif

/**
*   Copyright 2012,2015 Keith Daigle
*   This file is part of cuda-bwt.
*
*   cuda-bwt is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   cuda-bwt is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with cuda-bwt.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef BUCKET_T_H
#define BUCKET_T_H
//This is the definition of the bucket which is primarily used to compare
//entries for each rotation
#define BUCKET_T_DATA_SIZE (sizeof(unsigned long long int) * 2)
typedef struct dBucket{
	unsigned int bIndex;
        unsigned long long int high;
        unsigned long long int low;

  __host__ __device__
    bool operator<(const dBucket& other) const
    {
	//Compare the bucket index first, if they don't match
	//just return the value of that comparison
	if(other.bIndex == bIndex)
	{
        	if(other.high == high)
        		return low < other.low ;
        	else
                	return high < other.high;
	}
	return bIndex < other.bIndex;
    }
  __host__ __device__
    bool operator==(const dBucket& other) const
    {
//        if(other.bIndex == bIndex && other.high == high && other.low == low)
        if(other.high == high && other.low == low)
                return true;
        else
                return false;
    }
} bucket_t;
#endif

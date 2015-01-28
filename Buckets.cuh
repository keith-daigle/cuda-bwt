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
#include "bucket.h"
//This kernel creates buckets for bwt rotations, the total number of bytes (chars) should be at least 16 in array s
//in the input s
//s - source data array of chars
//bucket - output array containing buckets
//rot_num - is the rotation for which this bucket is created
//offset - is the number of buckets to the left from our rot_num that this bucket will be made for
//exclude - is a matches array as created by the Matches kernel, if index for this bucket is n+1 we skip generation of next bucket
//n - is the number of elements to create buckets for
//BLOCK_LEN - needs to be passed in so that the thread's index will be calcualted properly
__global__ void Buckets( unsigned char * s, bucket_t * bucket, unsigned int * rot_num, unsigned int * exclude, unsigned int offset, unsigned int n, unsigned int BLOCK_LEN)
{

	unsigned int i = (blockIdx.x * BLOCK_LEN) + threadIdx.x;
	//there's no appreciable speed difference  between the modulus version of the code
	//and non modulus version of the code based upon tests
	if (i < n)
	{	
		uint64_t a;
		uint64_t b;
		//register bucket_t t;
		register unsigned int bucket_start;
		//the offset tells us how many buckets over from this starting index
		//we should be generating the next bucket for, note that this is a offset in
		//the number of buckets, not bytes
		if(offset)
			//bucket_start = rot_num[i] + (sizeof(bucket_t) * offset);
			bucket_start = rot_num[i] + (BUCKET_T_DATA_SIZE * offset);
		else
			bucket_start = i;
		//we use the n+1 value to tell us that we should not be generating another
		//bucket for this index to hopefully save time as more rows of the
		//rotation matrix become sorted
		if( exclude[i] != n+1 )
		{
			a=0;
			b=0;
			for(int j = 0; j<8 ; j++)
			{
				a = a<<8;
				a |= s[(bucket_start+j) % n];
				b = b<<8;
				b |= s[(bucket_start+j+8) % n];
			}
			bucket[i].high = a;
			bucket[i].low = b;
		}

		//if we're running this the first time we just assign the
		//rotation number which is really this thread's index
		//this really should be done with a counting iterator
		if(offset == 0)
		{
			bucket[i].bIndex = 0;
			rot_num[i] = i;
		}
	}
	__syncthreads();
}

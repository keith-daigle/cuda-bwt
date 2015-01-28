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
//This kernel will walk through an array and group buckets to be sorted together
//the bucket_index array will contain the start index in the array of each bucket, and each subsequent member of that bucket
//will have it's index in the bucket_index array set to the same number.
//Entries that are not a part of a bucket have their index in the bucket_index array set to  n+1
//the touched array can be used to make sure that we're actually walking through those indexes to update the bucket index
//__global__ void InnerWalkMatches( bucket_t * buckets, unsigned int * bucket_index, unsigned int * prev_bucket_index, unsigned int * touched, unsigned int n,  unsigned int BLOCK_LEN)
__global__ void InnerWalkMatches( bucket_t * buckets,
		unsigned int * bucket_index,
		unsigned int * prev_bucket_index,
		unsigned int n,
	       	unsigned int BLOCK_LEN)
{
        unsigned int i = (blockIdx.x * BLOCK_LEN) + threadIdx.x;
        unsigned int prev_bucket = ((i+n)-1) % n;
        unsigned int next_bucket = ((i+n)+1) % n;
        if( i < n )
        {
		if(prev_bucket_index[i] == n+1)
		{
			bucket_index[i] = n+1;
			buckets[i].bIndex = n+1;
		}
		//so if we are at the left edge of a bucket that existed previously
		//and our index isn't n+1, try and build the bucket index for this bucket
		else if( prev_bucket_index[i] == prev_bucket_index[next_bucket] &&
				//this attempts to catch case where the previous run through some of 
				//this indexe's bucket was sorted away, since then we ignore the condition where
				//we look for an edge in the previous buckets and instead look for edge in current buckets
		       ( prev_bucket_index[i] != prev_bucket_index[prev_bucket] ||
		       ( prev_bucket_index[i] == prev_bucket_index[prev_bucket] && bucket_index[prev_bucket] == n+1) )&&
		       bucket_index[i] != n+1)
		{
			//so while we're still within the previously defined bucket
			//test to make sure that the bucket value still matches the next one
			register unsigned int current_bucket_idx = i;
			register unsigned int j = i;
			while( prev_bucket_index[i] == prev_bucket_index[j])
			{
				while(buckets[current_bucket_idx] == buckets[j] &&
				      prev_bucket_index[i] == prev_bucket_index[j])
				{
//					touched[j]++;
					bucket_index[j] = current_bucket_idx;
					buckets[j].bIndex = current_bucket_idx;
					j++;
				}
				current_bucket_idx = j;
			}
		}
	}
}

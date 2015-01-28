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
__global__ void WalkMatches( bucket_t * buckets, unsigned int * bucket_index, unsigned int * prev_bucket_index, unsigned int n, unsigned int BLOCK_LEN)
{
        unsigned int i = (blockIdx.x * BLOCK_LEN) + threadIdx.x;
        unsigned int prev_bucket = ((i+n)-1) % n;
        if( i < n )
        {
                //we want to start on our bucket and walk through it
                //verify that we are starting a bucket by checking the value of the previous bucket
		//to see if it differs from this bucket
		//this may fail in the case of 2 adjacent buckets ad different points in the program
		//just happen to be equal, I need another way of handling this
                if( bucket_index[i] != n+1 && i < n && !( buckets[i] == buckets[prev_bucket] ) )
                {
                        unsigned int current_bucket_idx = i;
                        unsigned int j = i+1;
                        while(buckets[i] == buckets[j] && bucket_index[j] != n+1 )
                        {
                                //this tests to ensure that we don't cross over any
                                //inter-bucket boundaries
                                if(prev_bucket_index[j] != prev_bucket_index[i])
                                {
                                        //not sure that this is necessary
                                        //since the already sorted indexes should be in bucket_index
                                        if(prev_bucket_index[j] == n+1)
                                                break;
                                        else
					{
                                                current_bucket_idx = j;
					}
                                }
				buckets[j].bIndex = current_bucket_idx;
                                bucket_index[j] = current_bucket_idx;
				j++;
                        }
                }
        }
}


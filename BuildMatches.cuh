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
__global__ void BuildMatches( bucket_t * buckets, unsigned int * bucket_index, unsigned int * prev_bucket_index, unsigned int n, unsigned int BLOCK_LEN)
{
        unsigned int i = (blockIdx.x * BLOCK_LEN) + threadIdx.x;
        unsigned int prev_bucket = ((i+n)-1) % n;
        unsigned int next_bucket = ((i+n)+1) % n;
        if( i < n )
        {
                //we want to pull in the information from the previous set of bucket
                //indexes about which buckets were already sorted out
                if(prev_bucket_index[i] == n+1)
                {
                        bucket_index[i] = n+1;
			buckets[i].bIndex = n+1;
                }
                //otherwise mark our index and our neghibor's index if the bucket to the left matches
                else
                {

                        if( i>0 && buckets[prev_bucket] == buckets[i] )
                        {
                                bucket_index[i] = i;
                              bucket_index[prev_bucket] = prev_bucket;
			      buckets[prev_bucket].bIndex = prev_bucket;
			      buckets[i].bIndex=i;
                        }
                        //so if the previous bucket didn't match, test the next bucket
                        //if it doesn't match too, mark this index as being done
			//the original compare was to n-1 index, but that left one possible
			//index at the very end as unmarked if it's different on it's own
			//the ends could probably also be handled in an else-if if the <n 
			//causes other problems
                        //else if( i< n-1)
                        else if( i< n)
                        {
                              if(! (buckets[next_bucket] == buckets[i] ) )
			      {
                                        bucket_index[i] = n+1;
					buckets[i].bIndex=n+1;
			      }
                        }
                }
                __syncthreads();
                //At this point, we want to see if any of our bucket indexes are bordered on the left and right
                //by a n+1, because if they are, this bucket is sorted by the fact of it's neghibors beingsorted
                if(bucket_index[prev_bucket] == n+1 && bucket_index[next_bucket] == n+1)
                {
                          bucket_index[i] = n+1;
			  buckets[i].bIndex=n+1;
                }
                
        }
}


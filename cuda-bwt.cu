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

#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>

#include "bucket.h"
#include "Buckets.cuh"
#include "BuildMatches.cuh"
#include "WalkMatches.cuh"
#include "InnerWalkMatches.cuh"
#include "CalcPushPositions.cuh"
#include "DoPush.cuh"
#include "timers.cuh"
#include "verifier.h"

//block size has a noticable effect on the speed of matches and buckets
//kernels, depends on device's abilities
//#define BLOCK_LEN 1024
#define BLOCK_LEN 512
//#define BLOCK_LEN 256
//This comparison is used to allow values to be scattered and gathered
//it should be set to the maximum input length, and compared to the bucket value
struct stencilComp {

	unsigned int pass;
        stencilComp( unsigned int limit) : pass(limit) {}
        __host__ __device__
	bool operator() ( const unsigned int& val)
	{
		return (val < pass);
	}
};

struct stencilCompInv {

	unsigned int pass;
        stencilCompInv( unsigned int limit) : pass(limit) {}
        __host__ __device__
	bool operator() ( const unsigned int& val)
	{
		return (val >= pass);
	}
};
int main( int argc, char * argv[])
{
  unsigned char * raw_input_ptr = NULL;
  bucket_t * bucket_ptr = NULL;
  unsigned int * rotation_idx_ptr = NULL;
  unsigned int * pbucket_idx_ptr = NULL;
  unsigned int * bucket_idx_ptr = NULL;
  unsigned char * input = NULL; 
  unsigned int * sorted_rotations;
  if(argc != 2 )
  {
	std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
	return 1;
  }
  std::cerr << "Reading file: " << argv[1] << std::endl;
  std::ifstream inFile( argv[1], std::ifstream::in|std::ifstream::binary);
  if(!inFile.is_open())
  {
	std::cerr << "Couldn't open file: " << argv[1] << std::endl;
	return 1;
  }
  //seek to the end of the file to get length, and return pointer to beginning
  inFile.seekg(0, std::ios_base::end);
  int inputLen = inFile.tellg();
  inFile.seekg(0, std::ios_base::beg);

  input = new unsigned char[inputLen];
  inFile.read(( char * ) input, inputLen);
  if( inFile.gcount() == inputLen)
	std::cerr<< "Read " << inputLen << " bytes from input file" << std::endl;
  else
  {
	std::cerr<< "Only read: " << inFile.gcount() << " bytes, expected " << inputLen << std::endl;
        delete[] input;
        return 1;
  }
  inFile.close(); 
  //These 2 comparisons are used by the count_if
  //and copy_if functions to use the bucket_indexes as a stencil
  //wherever bucket_index[i] = inputLen+1 the data will *NOT* be copied
  //so we use inputLen+1 to know when a index is in it's final sorted position
  stencilComp still_valid(inputLen);
  stencilCompInv doCopy(inputLen);

  //time the malloc and copy
  timer_start();

  //alloc and memcopy the input to the device
  cudaMalloc( &raw_input_ptr , inputLen * sizeof(char) );
  cudaMemcpy( raw_input_ptr, input, inputLen, cudaMemcpyHostToDevice);

  //memory for buckets which contain 16 bytes of each rotation
  cudaMalloc( &bucket_ptr , inputLen * sizeof(bucket_t) );
  thrust::device_ptr<bucket_t> buckets_on_device((bucket_t *)bucket_ptr);

  //memory for starting index of each rotation, this is what we're really interested in
  //after all the sorting
  cudaMalloc( &rotation_idx_ptr , inputLen * sizeof(unsigned int) );
  thrust::device_ptr<unsigned int> rotation_indexes_on_device((unsigned int *)rotation_idx_ptr);

  //This is the starting index of the current bucket 
  cudaMalloc( &bucket_idx_ptr, inputLen * sizeof(unsigned int) );
  thrust::device_ptr<unsigned int> bucket_indexes((unsigned int *)bucket_idx_ptr);

  //This is the starting index of the current bucket, from the previous iteration
  cudaMalloc( &pbucket_idx_ptr, inputLen * sizeof(unsigned int) );
  thrust::device_ptr<unsigned int> prev_bucket_indexes((unsigned int *)pbucket_idx_ptr);

  thrust::fill(prev_bucket_indexes, prev_bucket_indexes+inputLen, 0);
  thrust::fill(bucket_indexes, bucket_indexes+inputLen, 0);
  std::cerr << "time to malloc, copy, assign pointers, and zero (ms): " << timer_stop_and_display() << std::endl;

  dim3 block(BLOCK_LEN);
  dim3 grid((inputLen/BLOCK_LEN)+1);
  std::cerr << "calling buckets, sort, matches count_if ....." << std::endl;
  timer_start();
  //Generate our buckets with the rotation indexes, the kernel handles populating the rotation_idx_ptr in order
  //by using the thread's index
  Buckets<<<grid, block>>>( raw_input_ptr, bucket_ptr, rotation_idx_ptr, bucket_idx_ptr , 0, inputLen, BLOCK_LEN);
  //now we take our first stab at sorting the buckets, to see how much we can shake out on the first run
  thrust::stable_sort_by_key(buckets_on_device, buckets_on_device+inputLen, rotation_indexes_on_device);

  //Now we organize any group of buckets and build the bucket_idx_ptr array to contain the starting
  //index of any such group of buckets
  BuildMatches<<<grid, block>>>( bucket_ptr, bucket_idx_ptr, pbucket_idx_ptr, inputLen, BLOCK_LEN);
  WalkMatches<<<grid, block>>>( bucket_ptr, bucket_idx_ptr, pbucket_idx_ptr, inputLen, BLOCK_LEN);
  int cnt = thrust::count_if( bucket_indexes, bucket_indexes+inputLen,  still_valid);

  //after the first round of sort, buckets and matches, we'll use however much 
  //a reduction the sort and matches yeilded to setup a area to scatter the data into
  //for sorting a smaller number of keys and values
  unsigned int * scattered_bucket_indexes;
  unsigned int * scattered_previous_bucket_indexes;
  unsigned int * scattered_rotation_indexes;
  bucket_t * scattered_buckets;

  //this will hold the reduced bucket indexes for sorting and comparing in loop
  cudaMalloc(&scattered_bucket_indexes, cnt * sizeof(unsigned int));
  thrust::device_ptr<unsigned int> scattered_bucket_indexes_on_device(scattered_bucket_indexes);

  //this will hold the previous set of reduced bucket indexes for sorting and comparing in loop
  cudaMalloc(&scattered_previous_bucket_indexes, cnt * sizeof(unsigned int));
  thrust::device_ptr<unsigned int> scattered_previous_bucket_indexes_on_device(scattered_previous_bucket_indexes);

  //this will hold the set of rotation indexes that are just along for the ride while sorting and comparing in loop
  cudaMalloc(&scattered_rotation_indexes, cnt * sizeof(unsigned int));
  thrust::device_ptr<unsigned int> scattered_rotation_indexes_on_device(scattered_rotation_indexes);

  //this will hold the buckets after being reduced in the loop, this is after generation
  //with Buckets kernel on full array of data
  cudaMalloc(&scattered_buckets, cnt * sizeof(bucket_t));
  thrust::device_ptr<bucket_t> scattered_buckets_on_device((bucket_t* )scattered_buckets);
  
  int k = 1;
// this sizeof(bucket_t) needs to get changed if i include
// the bucket number in the struct, probably need a BUCKET_BYTES define or something
  
  while( cnt > 0 && k < (inputLen/BUCKET_T_DATA_SIZE)+1 )
  {

    //First off, we generate the buckets to compare our current rotations that aren't marked as sorted
    //this uses k as an counter to step the bucket generation forward in each rotation to the next bucket
    Buckets<<<grid, block>>>( raw_input_ptr, bucket_ptr, rotation_idx_ptr, bucket_idx_ptr, k, inputLen, BLOCK_LEN);

    //Now, 'gather' the bucket value and index number along with rotation indexes into smaller arrays to be operated on
    thrust::remove_copy_if(thrust::make_zip_iterator(thrust::make_tuple(buckets_on_device, rotation_indexes_on_device, bucket_indexes)),
    			   thrust::make_zip_iterator(thrust::make_tuple(buckets_on_device+inputLen, rotation_indexes_on_device+inputLen, bucket_indexes+inputLen)),
			   bucket_indexes,
			   thrust::make_zip_iterator(thrust::make_tuple(scattered_buckets_on_device, scattered_rotation_indexes_on_device, scattered_bucket_indexes_on_device)),
			   doCopy);
     
    //Next the sort, we  use the bucket index (bIndex in bucket_t) first, then the bucket value.  The stable sort on the index helps keep them grouped together
	thrust::stable_sort_by_key(scattered_buckets_on_device, scattered_buckets_on_device+cnt, 
	  	thrust::make_zip_iterator(
		thrust::make_tuple(scattered_bucket_indexes_on_device, scattered_rotation_indexes_on_device) ) );

    //After the sorting is done, we will figure out where things are to be sent back to, in the grand scheme of things
    //CalcPushPositions calculates new scattered_bucket_indexes based upon the state of the reduced buckets and rotation indexes
    //DoPush simply pushes the data back to the main rotation_idx_ptr array and bucket_ptr array
    CalcPushPositions<<<grid, block>>>(bucket_ptr, rotation_idx_ptr, scattered_buckets, scattered_rotation_indexes, scattered_bucket_indexes, scattered_previous_bucket_indexes, cnt, BLOCK_LEN);
    DoPush<<<grid, block>>>(bucket_ptr, rotation_idx_ptr, scattered_buckets, scattered_rotation_indexes, scattered_bucket_indexes, scattered_previous_bucket_indexes, cnt, BLOCK_LEN);

    //We now copy the bucket indexes to the previous set of bucket indexes in order to use the data
    //to build our bucket indexes
    cudaMemcpy(pbucket_idx_ptr, bucket_idx_ptr, inputLen*sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    BuildMatches<<<grid, block>>>( bucket_ptr, bucket_idx_ptr, pbucket_idx_ptr, inputLen, BLOCK_LEN);
    InnerWalkMatches<<<grid, block>>>( bucket_ptr, bucket_idx_ptr, pbucket_idx_ptr, inputLen, BLOCK_LEN);

    //lastly, we count the number of buckets still to be sorted to see if we can stop looping early
    cnt = thrust::count_if( bucket_indexes, bucket_indexes+inputLen,  still_valid) ;
    k++;
  }
  std::cerr << "Total time taken is: " << timer_stop_and_display() << " (ms)" << std::endl;


  sorted_rotations = new unsigned int[inputLen];
  cudaMemcpy( sorted_rotations, rotation_idx_ptr, inputLen*sizeof(unsigned int), cudaMemcpyDeviceToHost);


  char * transformedData = new char[inputLen];
  unsigned int firstIndex = 0;
  //walk through our sorted rotations, and take the 
  //last character from each rotation for the bwt
  std::cerr << "collecting output into array.." << std::endl;
  for( unsigned int l = 0; l < inputLen; l++)
  {
     if(sorted_rotations[l] == 0)
     {
	//this is the pointer to the end of the string so that
	//when someone goes to reconstruct the original, they can use this
	//to know where the end is
	firstIndex = l;
	transformedData[l] = input[inputLen -1];
     }
     else
     {
	transformedData[l] = input[sorted_rotations[l]-1];
     }
  }
  std::cerr << "writing output file.." << std::endl;
  std::string ofname(argv[1]);
  ofname+=".enc";
  std::ofstream outFile( ofname.c_str(), std::ifstream::out|std::ifstream::binary);
  outFile.write((char *) &firstIndex, sizeof(unsigned int));
  outFile.write(transformedData, inputLen);
  outFile.close();

  cudaFree( raw_input_ptr );
  cudaFree( scattered_previous_bucket_indexes );
  cudaFree( rotation_idx_ptr );
  cudaFree( bucket_idx_ptr );
  cudaFree( scattered_bucket_indexes );
  cudaFree( scattered_previous_bucket_indexes );
  cudaFree( scattered_rotation_indexes );
  cudaFree( scattered_buckets );
  delete[] input;

return 0;
}

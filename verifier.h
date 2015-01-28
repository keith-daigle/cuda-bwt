#include <iostream>
#include <cuda_runtime.h>
#include "bucket.h"

void verify_rotation_indexes( unsigned int n, void * nts_one)
{
  unsigned int * uno;
  uno = new unsigned int[n];

  std::cout << "attempting to copy back: " << std::dec << n << " rotation indexes to uno.. " << std::endl;
  cudaMemcpy( uno, nts_one, n*sizeof(unsigned int), cudaMemcpyDeviceToHost);

  for( int i = 0; i<n; i++)
  {
	  for( int j = 0; j < n; j++ )
	  {
            if( uno[i]  == uno[j]  && i != j )
            	std::cout << "indexes: " << std::dec << i << " with value " << uno[i] 
			  << " and " << std::dec << j  << " with value " << uno[j]
			  << " out of: " << std::dec << n << " match!" << std::endl;
	  }
  }
  delete[] uno;
}
void verify_rotation_indexes_buckets_bucket_indexes( unsigned int n, void * nts_one, void * buckets, void * bucket_indexes)
{
  unsigned int * ri;
  bucket_t  * b;
  unsigned int * bi;
  ri = new unsigned int[n];
  b = new bucket_t[n];
  bi = new unsigned int[n];

  std::cout << "attempting to copy back: " << std::dec << n << " rotation indexes to ri.. " << std::endl;
  cudaMemcpy( ri, nts_one, n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy( b, buckets, n*sizeof(bucket_t), cudaMemcpyDeviceToHost);
  cudaMemcpy( bi, bucket_indexes, n*sizeof(unsigned int), cudaMemcpyDeviceToHost);

  for( int i = 0; i<n; i++)
  {
        std::cout << std::endl << "bucket on index: " << std::dec << i << " starting at: ";
	std::cout << std::setw(9) << std::setfill('0') << std::dec << bi[i] << " : ";
        std::cout << std::setw(16) << std::setfill('0') << std::hex << b[i].high << " : ";
        std::cout << std::setw(16) << std::setfill('0') << std::hex << b[i].low;
	for( int j = 0; j < n; j++ )
	{
          if( ri[i]  == ri[j]  && i != j )
          	std::cout << " <--- indexes: " << std::dec << i << " with value " << ri[i] 
			  << " and " << std::dec << j  << " with value " << ri[j]
			  << " out of: " << std::dec << n << " match!";
	}
  }
  std::cout << std::endl;
  delete[] ri;
  delete[] b;
  delete[] bi;
}
void verify_rotation_indexes_buckets( unsigned int n, unsigned int k, void * rotation_indexes, void * buckets, unsigned char * const raw_input)
{
  unsigned int * ri;
  bucket_t  * b;
  ri = new unsigned int[n];
  b = new bucket_t[n];

  std::cout << "attempting to copy back: " << std::dec << n << " rotation indexes to ri.. " << std::endl;
  cudaMemcpy( ri, rotation_indexes, n*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy( b, buckets, n*sizeof(bucket_t), cudaMemcpyDeviceToHost);

  for( int w = 0; w<n; w++)
  {

      for(int x=0; x<sizeof(unsigned long long int); x++)
      {
	   //check if this bucket's values match the values
	   //need to start at this rotation index, plus k*number of bytes in bucket
	   //then walk backwards to make sure we compare bucket correctly
	   if( (( b[w].high >> x*8 ) & 0x00000000000000ff) != raw_input[((ri[w] + (k * sizeof(bucket_t)) ) + sizeof(unsigned long long int) - (x+1))%n ])
	   {
             std::cout << std::endl << "high bucket on index: " << std::setw(9) << std::setfill('0') << std::dec << w << " starting at: ";
	     std::cout << std::setw(9) << std::setfill('0') << std::dec << ri[w] << " moved out " << std::dec << k << " places mismatch at place " << x << " - ";
	     std::cout << std::setw(2) << std::setfill('0') << std::hex << (( b[w].high >> x*8 ) & 0x00000000000000ff) << " : ";
	     std::cout << std::setw(2) << std::setfill('0') << std::hex << (unsigned int) raw_input[((ri[w] + (k * sizeof(bucket_t)) ) + sizeof(unsigned long long int) - (x+1))%n ] << " : " ;
             std::cout << std::setw(16) << std::setfill('0') << std::hex << b[w].high  << " : ";
		for(int y = 0; y< sizeof(unsigned long long int); y++)
			std::cout << std::setw(2) << std::setfill('0') << std::hex << (int) raw_input[((ri[w] + (k * sizeof(bucket_t)) ) + y)%n ];
	   }
	   if( (( b[w].low >> x*8 ) & 0x00000000000000ff) != raw_input[((ri[w] + (k * sizeof(bucket_t)) ) + (sizeof(unsigned long long int)*2) - (x+1))%n ])
	   {
             std::cout << std::endl << "low  bucket on index: " << std::setw(9) << std::setfill('0') << std::dec << w << " starting at: ";
	     std::cout << std::setw(9) << std::setfill('0') << std::dec << ri[w] << " moved out " << std::dec << k << " places mismatch at place " << x << " - ";
	     std::cout << std::setw(2) << std::setfill('0') << std::hex << (( b[w].low >> x*8 ) & 0x00000000000000ff) << " : ";
	     std::cout << std::setw(2) << std::setfill('0') << std::hex << (unsigned int) raw_input[((ri[w] + (k * sizeof(bucket_t)) ) + (sizeof(unsigned long long int)*2) - (x+1))%n ] << " : ";
             std::cout << std::setw(16) << std::setfill('0') << std::hex <<  b[w].low << " : ";
		for(int y = 0; y< sizeof(unsigned long long int); y++)
			std::cout << std::setw(2) << std::setfill('0') << std::hex << (int) raw_input[((ri[w] + (k * sizeof(bucket_t)) ) + (sizeof(unsigned long long int) + y))%n ];
	   }
      }
  }
  std::cout << std::endl;
  delete[] ri;
  delete[] b;
}

void dump_buckets(unsigned int n, void * bucket_ptr)
{
  bucket_t * bucks;
  bucks = new bucket_t[n];
  std::cout << "attempting to copy back: " << std::dec << n << " buckets... " << std::endl;
  cudaMemcpy( bucks, bucket_ptr, n*sizeof(bucket_t), cudaMemcpyDeviceToHost);
  for( int i = 0; i<n; i++)
  {
        std::cout << "bucket on index: " << std::dec << i << " : " ;
        std::cout << std::setw(16) << std::setfill('0') << std::hex << bucks[i].high << " : ";
        std::cout << std::setw(16) << std::setfill('0') << std::hex << bucks[i].low << std::endl;
  }
  delete[] bucks;
}

void dump_buckets_matches( unsigned int n, void * bucket_ptr, void * matches_ptr)
{
  bucket_t * bucks;
  unsigned int * matches;
  bucks = new bucket_t[n];
  matches = new unsigned int[n];

  std::cout << "attempting to copy back: " << std::dec << n << " buckets.. " << std::endl;
  cudaMemcpy( bucks, bucket_ptr, n*sizeof(bucket_t), cudaMemcpyDeviceToHost);

  std::cout << "attempting to copy back: " << std::dec << n << " matches.. " << std::endl;
  cudaMemcpy( matches, matches_ptr, n*sizeof(unsigned int), cudaMemcpyDeviceToHost);

  for( int i = 0; i<n; i++)
  {
        std::cout << "bucket on index: " << std::dec << i << " matches buckets starting at: ";
        std::cout << std::setw(9) << std::setfill('0') << std::dec << matches[i] << " : ";
        std::cout << std::setw(16) << std::setfill('0') << std::hex << bucks[i].high << " : ";
        std::cout << std::setw(16) << std::setfill('0') << std::hex << bucks[i].low << std::endl;
  }
  delete[] bucks;
  delete[] matches;
}

void dump_buckets_withindex_matches( unsigned int n, void * bucket_ptr, void * matches_ptr)
{
  bucket_t * bucks;
  unsigned int * matches;
  bucks = new bucket_t[n];
  matches = new unsigned int[n];

  std::cout << "attempting to copy back: " << std::dec << n << " buckets.. " << std::endl;
  cudaMemcpy( bucks, bucket_ptr, n*sizeof(bucket_t), cudaMemcpyDeviceToHost);

  std::cout << "attempting to copy back: " << std::dec << n << " matches.. " << std::endl;
  cudaMemcpy( matches, matches_ptr, n*sizeof(unsigned int), cudaMemcpyDeviceToHost);

  for( int i = 0; i<n; i++)
  {
        std::cout << "bucket on index: " << std::dec << i << " matches buckets starting at: ";
        std::cout << std::setw(9) << std::setfill('0') << std::dec << matches[i] << " : ";
        std::cout << std::setw(16) << std::setfill('0') << std::hex << bucks[i].high << " : ";
        std::cout << std::setw(16) << std::setfill('0') << std::hex << bucks[i].low << " : " ;
	std::cout << std::setw(9) << std::setfill('0') << std::dec << bucks[i].bIndex << std::endl;
  }
  delete[] bucks;
  delete[] matches;
}
void dump_bucket_prev_curr( unsigned int n, void * bucket_ptr, void * matches_ptr, void * prev_bucket_ptr, void * prev_matches_ptr)
{
  bucket_t * bucks;
  bucket_t * pbucks;
  unsigned int * matches;
  unsigned int * pmatches;
  bucks = new bucket_t[n];
  pbucks = new bucket_t[n];
  matches = new unsigned int[n];
  pmatches = new unsigned int[n];

  std::cout << "attempting to copy back: " << std::dec << n << " buckets.. " << std::endl;
  cudaMemcpy( bucks, bucket_ptr, n*sizeof(bucket_t), cudaMemcpyDeviceToHost);

  std::cout << "attempting to copy back: " << std::dec << n << " buckets.. " << std::endl;
  cudaMemcpy( pbucks, prev_bucket_ptr, n*sizeof(bucket_t), cudaMemcpyDeviceToHost);

  std::cout << "attempting to copy back: " << std::dec << n << " matches.. " << std::endl;
  cudaMemcpy( matches, matches_ptr, n*sizeof(unsigned int), cudaMemcpyDeviceToHost);

  std::cout << "attempting to copy back: " << std::dec << n << " matches.. " << std::endl;
  cudaMemcpy( pmatches, prev_matches_ptr, n*sizeof(unsigned int), cudaMemcpyDeviceToHost);

  for( int i = 0; i<n; i++)
  {
        std::cout << "bucket on index: " << std::dec << i << " prev: ";
        std::cout << std::setw(9) << std::setfill('0') << std::dec << pmatches[i] << " : ";
        std::cout << std::setw(16) << std::setfill('0') << std::hex << pbucks[i].high << " : ";
        std::cout << std::setw(16) << std::setfill('0') << std::hex << pbucks[i].low << " curr: " ;
        std::cout << std::setw(9) << std::setfill('0') << std::dec << matches[i] << " : ";
        std::cout << std::setw(16) << std::setfill('0') << std::hex << bucks[i].high << " : ";
        std::cout << std::setw(16) << std::setfill('0') << std::hex << bucks[i].low << std::endl;
  }
  delete[] bucks;
  delete[] matches;
}


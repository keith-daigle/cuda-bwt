# Burrows-Wheeler Transform (BWT) for CUDA

## Synopsis
The Burrows-Wheeler Transform forms the core of bzip2 compression algorithm. The original [research paper](http://www.hpl.hp.com/techreports/Compaq-DEC/SRC-RR-124.pdf) is sitll available from HP Labs by way of DEC.  In essence it's a very clever way of ordering a set of data to reveal patterns that make the data more easily compressible.  This implementation was a project I worked on while finishing up my undergrad degree.  It was done for a poster session, which it won in it's category.  I credit that to partially being a visually attractive poster (included in repo) and partially thanks to my dogged will to explain the transform to anyone who made eye contact.  It took quite a bit of work to get to the point where it would outperform a modern CPU on small to medium sized blocks (~1MB), but it utterly destroys a CPU on blocks larger than 2 or 3 MB.  In fact, I found it could handle very, very large blocks due to the way it slices up the rotations into buckets.


## Code
###Background
I had initially tried a naive approach to doing this with the classic pointer into the rotations with a sort, but that fails miserably on a GPU.   I found that naive approach on a GTX 480 was much slower than even a low end Core 2 processor.  The approach in the current code follows a slightly different algorithm.  It attempts to take slices of each rotation into a bucket and sort against them in a group of steps.  It keeps things together by marking grouped indexes together and using that in the sort, too.  On smaller blocks the memcpy time really hurts this implementation but it can still be competitive in some cases.  The larger the block the better it's performance when compared with a CPU.  I'd imagine that it has something to do with the amount of cache your CPU has available where the breakover occurs.

###Layout
The primary algorithm is in the cuda-bwt.cu file.  The main loop is pretty small in terms of # of calls, though the use of the zip iterators makes it look a little larger than it is.  The buckets.h file defines the structure for a single bucket.  It can be thought of as a slice of the current input rotations that are to be sorted.  There's a set of buckets that gets created each time through the main loop.  Any index which is not sorted will have a bucket created so the sorting can continue.  The Buckets.cuh file contains the code to build the buckets on the GPU from the previously memcpy'd source data.  The generated buckets are then sorted using thrust and some zip iterators to hold the rotation indexes.  The CalcPushPositions and DoPush are used to move the buckets and indexes that need sorting into smaller arrays. This is done for quicker sorts and the move the results back into the main data set.  The same goes for the WalkMatches and InnerWalkMatches, except they're used to flag buckets that need to be kept together so that the partially sorted data will stay together correctly. The verifier.h has some code that's useful for checking the data or the buckets while the transform is running. On large runs using the calls in that header produces way too much data to be handled by a human, but is still sometimes useful.

###Enhancements
There's probably some low hanging fruit in terms of using streams on different initial buckets to better overlap the data.  Nvvp is suggesting to do away with the count_if to avoid the device to host memcpy on each iteration.  Also as this was written against CUDA 3.5 and there's probably some better ways of handling it now.  For example maybe the unified memory model could be leveraged for better performance.  The code could probably be made a little more readable if I'd just use the namespaces.  There's almost certainly bugs in it that I haven't found, so please don't hesitate to contact me if you think you've found one.

## Prerequisites

A relatively modern CUDA environment is needed.  Generally CUDA 4.0 or better should be fine.  I've recently tested on 5.0 and 6.5 so newer versions work just as well.

## Installation

Clone the repo,  Make sure your environment is setup for CUDA ($PATH/$LD_LIBRARY_PATH) and do 'nvcc cuda-bwt.cu'.  It should compile it and leave you with a binary in a.out.

## Running

The a.out binary will take a single argument for it's input file.  It will transform the input file and write it along with the index of the original rotation to an output file which is the same name as the input file appended with .enc.

## Contributions

Anyone wanting who has questions can drop me an email.  If you do improve upon the current implementation, please do share.  I'll gladly accept a pull request, patches, or even lines of code by email.

## Why?
My curiosity surrounding the BWT goes back a ways as I was waiting for some files to compress and wondered WTF it was that bzip2 was doing that took it so long (but worked oh so well).  So I profiled it and was amazed that there was a sort in there and after wondering why, down the rabbit hole I went.  Many, many, many years later I still sit and think in awe that Burrows and Wheeler ever figured out that sorted rotations would end up revealing patterns. This is my cup of water thrown into their lake.


## License

    This code is licensed under the terms of the GNU GPL v3.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

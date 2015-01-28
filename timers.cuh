//This file was stolen from code posted on the web somewhere
//I can't recall where I got it, but it's not really a core part of the cuda-bwt


/********************** TIMER Functions *************/
//Global vars
cudaEvent_t start;
cudaEvent_t end;
void timer_start()
{
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start,0);
}

float timer_stop_and_display()
{
  float elapsed_time;
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, start, end);

  //std::cout << "  ( "<< elapsed_time << "ms )" << std::endl;

  return elapsed_time;
}
/********************** TIMER Functions *************/


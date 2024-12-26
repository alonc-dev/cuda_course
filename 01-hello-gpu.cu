#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  printf("%d) ThIdx.x=%d, BlkIdx.x=%d  - ",i, threadIdx.x, blockIdx.x);
  
  if(blockIdx.x == 5 && threadIdx.x == 3)
  {
    printf("Success!\n");
  } else {
    printf("Failure.\n");
  }
}

int main()
{
  /*
   * Update the execution configuration so that the kernel
   * will print `"Success!"`.
   */

  printSuccessForCorrectExecutionConfiguration<<<6, 4>>>();
  cudaDeviceSynchronize();

}

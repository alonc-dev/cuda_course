#include <stdio.h>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void addArraysInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  printf("%d) ThIdx.x=%d, BlkIdx.x=%d, BlkDim=%d, stride=%d \n",index, threadIdx.x, blockIdx.x, blockDim.x, stride);

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  printf("Device ID: %d\tNumber of SMs: %d\n", deviceId, numberOfSMs);

  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;

  /*
   * Grid sizes that are multiples of the number of available SMs can
   * increase performance.
   */

  numberOfBlocks = 32 * numberOfSMs;

  cudaError_t addArraysErr;
  cudaError_t asyncErr;

  printf("numberOfBlocks=%d,  threadsPerBlock=%d, N=%d\n",numberOfBlocks, threadsPerBlock, N);
  addArraysInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  addArraysErr = cudaGetLastError();
  if(addArraysErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addArraysErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

/*
 results:
 Device ID: 0    Number of SMs: 80
numberOfBlocks=2560,  threadsPerBlock=256, N=33554432
23232) ThIdx.x=192, BlkIdx.x=90, BlkDim=256, stride=655360 
23233) ThIdx.x=193, BlkIdx.x=90, BlkDim=256, stride=655360 
23234) ThIdx.x=194, BlkIdx.x=90, BlkDim=256, stride=655360 
23235) ThIdx.x=195, BlkIdx.x=90, BlkDim=256, stride=655360 
23236) ThIdx.x=196, BlkIdx.x=90, BlkDim=256, stride=655360 
23237) ThIdx.x=197, BlkIdx.x=90, BlkDim=256, stride=655360 
23238) ThIdx.x=198, BlkIdx.x=90, BlkDim=256, stride=655360 
23239) ThIdx.x=199, BlkIdx.x=90, BlkDim=256, stride=655360 
23240) ThIdx.x=200, BlkIdx.x=90, BlkDim=256, stride=655360 
23241) ThIdx.x=201, BlkIdx.x=90, BlkDim=256, stride=655360 
23242) ThIdx.x=202, BlkIdx.x=90, BlkDim=256, stride=655360 
23243) ThIdx.x=203, BlkIdx.x=90, BlkDim=256, stride=655360 
23244) ThIdx.x=204, BlkIdx.x=90, BlkDim=256, stride=655360 
23245) ThIdx.x=205, BlkIdx.x=90, BlkDim=256, stride=655360 
23246) ThIdx.x=206, BlkIdx.x=90, BlkDim=256, stride=655360 
23247) ThIdx.x=207, BlkIdx.x=90, BlkDim=256, stride=655360 
23248) ThIdx.x=208, BlkIdx.x=90, BlkDim=256, stride=655360 
23249) ThIdx.x=209, BlkIdx.x=90, BlkDim=256, stride=655360 
23250) ThIdx.x=210, BlkIdx.x=90, BlkDim=256, stride=655360 
23251) ThIdx.x=211, BlkIdx.x=90, BlkDim=256, stride=655360 
:
:
6144) ThIdx.x=0, BlkIdx.x=24, BlkDim=256, stride=655360 
6145) ThIdx.x=1, BlkIdx.x=24, BlkDim=256, stride=655360 
6146) ThIdx.x=2, BlkIdx.x=24, BlkDim=256, stride=655360 
6147) ThIdx.x=3, BlkIdx.x=24, BlkDim=256, stride=655360 
6148) ThIdx.x=4, BlkIdx.x=24, BlkDim=256, stride=655360 
6149) ThIdx.x=5, BlkIdx.x=24, BlkDim=256, stride=655360 
6150) ThIdx.x=6, BlkIdx.x=24, BlkDim=256, stride=655360 
6151) ThIdx.x=7, BlkIdx.x=24, BlkDim=256, stride=655360 
6152) ThIdx.x=8, BlkIdx.x=24, BlkDim=256, stride=655360 
6153) ThIdx.x=9, BlkIdx.x=24, BlkDim=256, stride=655360 
6154) ThIdx.x=10, BlkIdx.x=24, BlkDim=256, stride=655360 
6155) ThIdx.x=11, BlkIdx.x=24, BlkDim=256, stride=655360 
6156) ThIdx.x=12, BlkIdx.x=24, BlkDim=256, stride=655360 
6157) ThIdx.x=13, BlkIdx.x=24, BlkDim=256, stride=655360 
6158) ThIdx.x=14, BlkIdx.x=24, BlkDim=256, stride=655360 
6159) ThIdx.x=15, BlkIdx.x=24, BlkDim=256, stride=655360 
6160) ThIdx.x=16, BlkIdx.x=24, BlkDim=256, stride=655360 
6161) ThIdx.x=17, BlkIdx.x=24, BlkDim=256, stride=655360 
6162) ThIdx.x=18, BlkIdx.x=24, BlkDim=256, stride=655360 
6163) ThIdx.x=19, BlkIdx.x=24, BlkDim=256, stride=655360 
6164) ThIdx.x=20, BlkIdx.x=24, BlkDim=256, stride=655360 
6165) ThIdx.x=21, BlkIdx.x=24, BlkDim=256, stride=655360 
6166) ThIdx.x=22, BlkIdx.x=24, BlkDim=256, stride=655360 
6167) ThIdx.x=23, BlkIdx.x=24, BlkDim=256, stride=655360 
6168) ThIdx.x=24, BlkIdx.x=24, BlkDim=256, stride=655360 
6169) ThIdx.x=25, BlkIdx.x=24, BlkDim=256, stride=655360 
6170) ThIdx.x=26, BlkIdx.x=24, BlkDim=256, stride=655360 
6171) ThIdx.x=27, BlkIdx.x=24, BlkDim=256, stride=655360 
6172) ThIdx.x=28, BlkIdx.x=24, BlkDim=256, stride=655360 
6173) ThIdx.x=29, BlkIdx.x=24, BlkDim=256, stride=655360 
6174) ThIdx.x=30, BlkIdx.x=24, BlkDim=256, stride=655360 
6175) ThIdx.x=31, BlkIdx.x=24, BlkDim=256, stride=655360 
10720) ThIdx.x=224, BlkIdx.x=41, BlkDim=256, stride=655360 
10721) ThIdx.x=225, BlkIdx.x=41, BlkDim=256, stride=655360 
10722) ThIdx.x=226, BlkIdx.x=41, BlkDim=256, stride=655360 
10723) ThIdx.x=227, BlkIdx.x=41, BlkDim=256, stride=655360 
10724) ThIdx.x=228, BlkIdx.x=41, BlkDim=256, stride=655360 
10725) ThIdx.x=229, BlkIdx.x=41, BlkDim=256, stride=655360 
10726) ThIdx.x=230, BlkIdx.x=41, BlkDim=256, stride=655360 
10727) ThIdx.x=231, BlkIdx.x=41, BlkDim=256, stride=655360 
10728) ThIdx.x=232, BlkIdx.x=41, BlkDim=256, stride=655360 
10729) ThIdx.x=233, BlkIdx.x=41, BlkDim=256, stride=655360 
10730) ThIdx.x=234, BlkIdx.x=41, BlkDim=256, stride=655360 
10731) ThIdx.x=235, BlkIdx.x=41, BlkDim=256, stride=655360 
10732) ThIdx.x=236, BlkIdx.x=41, BlkDim=256, stride=655360 
10733) ThIdx.x=237, BlkIdx.x=41, BlkDim=256, stride=655360 
10734) ThIdx.x=238, BlkIdx.x=41, BlkDim=256, stride=655360 
10735) ThIdx.x=239, BlkIdx.x=41, BlkDim=256, stride=655360 
10736) ThIdx.x=240, BlkIdx.x=41, BlkDim=256, stride=655360 
10737) ThIdx.x=241, BlkIdx.x=41, BlkDim=256, stride=655360 
10738) ThIdx.x=242, BlkIdx.x=41, BlkDim=256, stride=655360 
10739) ThIdx.x=243, BlkIdx.x=41, BlkDim=256, stride=655360 
10740) ThIdx.x=244, BlkIdx.x=41, BlkDim=256, stride=655360 
10741) ThIdx.x=245, BlkIdx.x=41, BlkDim=256, stride=655360 
10742) ThIdx.x=246, BlkIdx.x=41, BlkDim=256, stride=655360 
10743) ThIdx.x=247, BlkIdx.x=41, BlkDim=256, stride=655360 
10744) ThIdx.x=248, BlkIdx.x=41, BlkDim=256, stride=655360 
10745) ThIdx.x=249, BlkIdx.x=41, BlkDim=256, stride=655360 
10746) ThIdx.x=250, BlkIdx.x=41, BlkDim=256, stride=655360 
10747) ThIdx.x=251, BlkIdx.x=41, BlkDim=256, stride=655360 
10748) ThIdx.x=252, BlkIdx.x=41, BlkDim=256, stride=655360 
:
:
*/


}

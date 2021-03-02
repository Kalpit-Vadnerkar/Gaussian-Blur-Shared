#include "./gaussian_kernel.h"



#define BLOCK 16
#define TILE_WIDTH 16

/*
The actual gaussian blur kernel to be implemented by 
you. Keep in mind that the kernel operates on a 
single channel.
 */
__global__ 
void gaussianBlur(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filterWidth){
      
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(c < cols && r < rows)
  {
    float pixelVal = 0.0f;
    
    for(int blurRow = 0; blurRow < filterWidth; ++blurRow)
    {
      for(int blurCol = 0; blurCol < filterWidth; ++blurCol)
      {
        int curRow = r + blurRow-filterWidth/2;
        int curCol = c + blurCol-filterWidth/2;

        int cR = max(0, min(rows-1, curRow));
        int cC = max(0, min(cols-1, curCol));
        
        if(cR < rows && cR > -1  && cC < cols && cC > -1)
        {
          pixelVal += (float)d_filter[blurRow * filterWidth + blurCol] * (float)d_in[cR * cols + cC];
        }
      }
    }
    
    d_out[r * cols + c] = (unsigned char)pixelVal;
  }


} 


__global__ 
void gaussianBlur_shared(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filterWidth){

  __shared__ unsigned char ds_in[TILE_WIDTH][TILE_WIDTH];
  __shared__ int p;

  float pixelVal = 0;

  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  

	// Load the tiles one by one

	p = c/TILE_WIDTH;
	// Check the boundary condition
	if((r < rows) && ((p*TILE_WIDTH+threadIdx.x) < cols)){

		// If the index is valid, load data to shared memory
		ds_in[threadIdx.y][threadIdx.x] = d_in[r*cols + p*TILE_WIDTH + threadIdx.x];
	}
	else{

		// If the index is invalid, load the zero pixel to shared memory
		ds_in[threadIdx.y][threadIdx.x] = (unsigned char)0.0;
	}

  __syncthreads();

  p = c/TILE_WIDTH;
	if ((r < rows) && ((p*TILE_WIDTH + threadIdx.x) < cols)) 
  {
    for(int blurRow = 0; blurRow < filterWidth; ++blurRow)
    {
      for(int blurCol = 0; blurCol < filterWidth; ++blurCol)
      {
        int curRow = threadIdx.y - (filterWidth/2);
        int curCol = threadIdx.x - (filterWidth/2);

        //curRow = min(max(curRow, 0), rows);
        //curCol = min(max(curCol, 0), cols);

        if(curRow >= -1 && curRow < rows && curCol >= -1 && curCol < cols)
        {
          pixelVal += d_filter[blurRow * filterWidth + blurCol] * (float)ds_in[curRow][curCol];
        }

        __syncthreads();
      }
    }
    
    d_out[r*cols+c] = (unsigned char)pixelVal;
	}

	// Barrier synchronization
	__syncthreads();
} 




__global__ 
void gaussianBlur_row(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filterWidth)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int leftCol = filterWidth/2;
  /*
  float rowsum = 0.0;
  for(int i = 0; i < filterWidth; i++)
  {
    rowsum += d_filter[leftCol*filterWidth+i];
  }
*/
  if(c < cols && r < rows)
  {
    float pixelVal = 0.0f;
    
    for(int blurRow = 0; blurRow < filterWidth; ++blurRow)
    {
        int curCol = c + blurRow - filterWidth/2;
        
        if(curCol > -1 && curCol < cols)
        {
         // printf("filter index: %d\n", blurRow*filterWidth+leftCol);
          //printf("d_in index: %d\n", r*cols+curCol);
          pixelVal += (float)d_filter[blurRow * filterWidth + leftCol] * (float)d_in[r * cols + curCol];
        }

      __syncthreads();
    }
    
    d_out[r * cols + c] = pixelVal;
  }

  __syncthreads();

}

__global__ 
void gaussianBlur_col(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filterWidth)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;

  int topRow = filterWidth/2;
  /*
  float rowsum = 0.0;
  for(int i = 0; i < filterWidth; i++)
  {
    rowsum += d_filter[i*filterWidth+topRow];
  }*/

  if(c < cols && r < rows)
  {
    float pixelVal = 0.0f;
    
    for(int blurCol = 0; blurCol < filterWidth; ++blurCol)
    {
        int curRow = r + blurCol - filterWidth/2;
        
        if(curRow > -1 && curRow < rows)
        {
          //printf("filter index: %d\n", topRow*filterWidth+blurCol);
          //printf("d_in index: %d\n", curRow*cols+c);
          pixelVal += ((float)d_filter[topRow * filterWidth + blurCol]) * (float)d_in[curRow * cols + c];
        }

      __syncthreads();
    }
    d_out[r * cols + c] = (unsigned char)pixelVal;
  }

  __syncthreads();

}




/*
  Given an input RGBA image separate 
  that into appropriate rgba channels.
 */
__global__ 
void separateChannels(uchar4 *d_imrgba, unsigned char *d_r, unsigned char *d_g, unsigned char *d_b,
        const int rows, const int cols){
        
        int c = blockIdx.x * blockDim.x + threadIdx.x;
	      int r = blockIdx.y * blockDim.y + threadIdx.y;
        int offset = r * cols + c;

        if(r < rows && c < cols)
        {
          d_r[offset] = d_imrgba[offset].x;
          d_g[offset] = d_imrgba[offset].y;
          d_b[offset] = d_imrgba[offset].z;
        }

} 
 

/*
  Given input channels combine them 
  into a single uchar4 channel. 

  You can use some handy constructors provided by the 
  cuda library i.e. 
  make_int2(x, y) -> creates a vector of type int2 having x,y components 
  make_uchar4(x,y,z,255) -> creates a vector of uchar4 type x,y,z components 
  the last argument being the transperency value. 
 */
__global__ 
void recombineChannels(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, uchar4 *d_orgba,
        const int rows, const int cols){
        
      int c = blockIdx.x * blockDim.x + threadIdx.x;
      int r = blockIdx.y * blockDim.y + threadIdx.y;
             
       if(r < rows && c < cols)
       {
          unsigned char red = d_r[r * cols + c];
          unsigned char green = d_g[r*cols+c];
          unsigned char blue = d_b[r*cols+c];
          
          uchar4 recombine = make_uchar4(blue, green, red, 255);
          
          d_orgba[r*cols+c] = recombine;
       } 


} 


void your_gauss_blur(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth){
 


        dim3 blockSize(BLOCK,BLOCK,1);
        dim3 gridSize(ceil(cols/BLOCK)+1,ceil(rows/BLOCK)+1,1);

        separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur_row<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter, filterWidth);
        gaussianBlur_col<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur_row<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter, filterWidth);
        gaussianBlur_col<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur_row<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter, filterWidth);
        gaussianBlur_col<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, rows, cols);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());   

}





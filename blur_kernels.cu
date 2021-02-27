#include "./gaussian_kernel.h" 


#define BLOCK 32
#define TILE_WIDTH BLOCK

/*
The actual gaussian blur kernel to be implemented by 
you. Keep in mind that the kernel operates on a 
single channel.
 */

//Shared memory kernel
__global__ 
void gaussianBlur_shared(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filterWidth){
  
  //shared memory array
  __shared__ unsigned char local_pixels[TILE_WIDTH + (filterWidth/2)*2][TILE_WIDTH + (filterWidth/2)*2];
  __shared__ float local_filter[filterWidth][filterWidth];
  
  // Global Image id
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  
  // Shared Data id
  int local_x = threadIdx.x + (filterWidth/2);
  int local_y = threadIdx.y + (filterWidth/2);

  
  // loading image into shared memory using the 1st thread.
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    for (int i=0; i < TILE_WIDTH + (filterWidth/2)*2; i++){
        for (int j=0; j < TILE_WIDTH + (filterWidth/2)*2; j++){
            if ((py + i - (filterWidth/2)) < rows && (py + i - (filterWidth/2)) > -1 && (px + j - (filterWidth/2)) < cols && (px + j - (filterWidth/2)) > -1){
                local_pixels[i][j] = d_in[(py + i - (filterWidth/2)) * numCols + (px + j - (filterWidth/2))];
            }
        }
    }
    // Loading the filter in the shared memory using the 1st thread.
    for (int i=0; i < filterWidth; i++){
        for (int j=0; j < filterWidth; j++){
            local_filter[i][j] = d_filter[i][j];
        }
    }
  }

  __syncthreads();

  if (px < cols && py < rows) {
    float pixval = 0.0;
    for(int blurRow = -(filterWidth / 2); blurRow < (filterWidth / 2) + 1; ++blurRow) {
        for(int blurCol = -(filterWidth / 2); blurCol < (filterWidth / 2) + 1; ++blurCol) {        
            int curRow = py + blurRow;
            int curCol = px + blurCol;
            if(curRow > -1 && curRow < rows && curCol > -1 && curCol < cols) {
                pixval += ((float) local_pixels[local_y + blurRow][local_x + blurCol] * local_filter[(blurRow + (filterWidth/2))][(blurCol + filterWidth/2)]);
                //pixval += ((float) d_in[curRow * cols + curCol] * d_filter[(blurRow + (filterWidth/2)) * filterWidth + (blurCol + filterWidth/2)]);
            }
        }
    }

    __syncthreads();
    d_out[py * cols + px] = (unsigned char) pixval;
  }
} 
















__global__ 
void gaussianBlur(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filterWidth){
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  int i = py * cols + px;
  if (px < cols && py < rows) {
    float pixval = 0.0;
    for(int blurRow = -(filterWidth / 2); blurRow < (filterWidth / 2) + 1; ++blurRow) {
        for(int blurCol = -(filterWidth / 2); blurCol < (filterWidth / 2) + 1; ++blurCol) {        
            int curRow = py + blurRow;
            int curCol = px + blurCol;
            if(curRow > -1 && curRow < rows && curCol > -1 && curCol < cols) {
                pixval += ((float) d_in[curRow * cols + curCol] * d_filter[(blurRow + (filterWidth/2))*filterWidth + (blurCol + filterWidth/2)]);
            }
        }
    }
    d_out[i] = (unsigned char) pixval;
  }
} 



/*
  Given an input RGBA image separate 
  that into appropriate rgba channels.
 */
__global__ 
void separateChannels(uchar4 *d_imrgba, unsigned char *d_r, unsigned char *d_g, unsigned char *d_b,
        const int rows, const int cols){

  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px < cols && py < rows) {
    int i = py * cols + px;
    d_r[i] = d_imrgba[i].x;
    d_g[i] = d_imrgba[i].y;
    d_b[i] = d_imrgba[i].z;
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
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px < cols && py < rows) {
    int i = py * cols + px;
    d_orgba[i] = make_uchar4(d_b[i], d_g[i], d_r[i], 255);
    //d_orgba[i].x = d_r[i];
    //d_orgba[i].y = d_g[i];
    //d_orgba[i].z = d_b[i];
  }
}

void your_gauss_blur(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth){
 


        dim3 blockSize(BLOCK, BLOCK, 1);
        dim3 gridSize((cols-1)/BLOCK + 1, (rows-1)/BLOCK + 1, 1);

        separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, rows, cols);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());   

}





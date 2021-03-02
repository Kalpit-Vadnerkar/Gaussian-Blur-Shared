#include "./gaussian_kernel.h" 


// Global memory block size
//#define BLOCK 32

// Shared memory tile size
#define TILE_WIDTH 24 
#define BLOCK TILE_WIDTH + 8

// Separable kernels block size
#define BLOCKCOL 16

/*
The actual gaussian blur kernel to be implemented by 
you. Keep in mind that the kernel operates on a 
single channel.
 */


// Separable Kernels

__global__ 
void gaussianBlur_separable_row(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filterWidth){
  
  //shared memory array
  __shared__ unsigned char local_pixels[BLOCK][BLOCK];
  __shared__ float local_filter[9][9];
  
  // id
  int px = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int py = blockIdx.y * TILE_WIDTH + threadIdx.y;
  
  // Shared Image id
  int local_x = threadIdx.x;
  int local_y = threadIdx.y;

  // Global Image id
  int global_x = px - filterWidth/2;
  int global_y = py - filterWidth/2;
  

  // Loading the image BLOCK in the shared memory.

  if (global_x > -1 && global_x < cols && global_y > -1 && global_y < rows){
    local_pixels[local_y][local_x] = d_in[global_y * cols + global_x];  
  }
  else
    local_pixels[local_y][local_x] = 0;
  

  // Loading the filter in the shared memory.
  
  if (threadIdx.x < filterWidth && threadIdx.y < filterWidth){
    local_filter[threadIdx.y][threadIdx.x] = d_filter[threadIdx.y * filterWidth + threadIdx.x];
  }

  __syncthreads();

  if (local_x < TILE_WIDTH && local_y < TILE_WIDTH && px < cols && py < rows) {
    float partial_pixval[9];
    for(int blurRow = -(filterWidth / 2); blurRow < (filterWidth / 2) + 1; ++blurRow) {
        partial_pixval[blurRow + (filterWidth / 2) - 1] = 0.0;
        for(int blurCol = -(filterWidth / 2); blurCol < (filterWidth / 2) + 1; ++blurCol) {        
            int curRow = local_y + blurRow + filterWidth / 2;
            int curCol = local_x + blurCol + filterWidth / 2;
            partial_pixval[blurRow + (filterWidth / 2) - 1] += (float) local_pixels[curRow][curCol] * local_filter[(blurRow + (filterWidth/2))][(blurCol + filterWidth/2)];
        }
    }
    __syncthreads();
    for (int i = 0; i < filterWidth; i++)
        d_out[(py * cols + px) * filterWidth + i] = (unsigned char) partial_pixval[i];
  }
} 

__global__
void gaussianBlur_separable_col(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filterWidth){
        
        int px = blockIdx.x * blockDim.x + threadIdx.x;
        int py = blockIdx.y * blockDim.y + threadIdx.y;
        
        int pix_number = py * cols + px;
        
        if (px < cols && py < rows){
            float sum = 0.0;
            for (int sum_i = 0; sum_i < filterWidth; sum_i++){
                sum += d_in[pix_number * filterWidth + sum_i];
            }
        __syncthreads();
        d_out[pix_number] = sum;
        }
}
















//Shared memory kernel

__global__ 
void gaussianBlur_shared(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, const int filterWidth){
  
  //shared memory array
  __shared__ unsigned char local_pixels[BLOCK][BLOCK];
  __shared__ float local_filter[9][9];
  
  // id
  int px = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int py = blockIdx.y * TILE_WIDTH + threadIdx.y;
  
  // Shared Image id
  int local_x = threadIdx.x;
  int local_y = threadIdx.y;

  // Global Image id
  int global_x = px - filterWidth/2;
  int global_y = py - filterWidth/2;
  
  // loading image into shared memory using the 1st thread.
  //if(threadIdx.x == 0 && threadIdx.y == 0) {
    //for (int i=0; i < TILE_WIDTH; i++){
      //  for (int j=0; j < TILE_WIDTH; j++){
        //    if ((py + i - (filterWidth/2)) < rows && (py + i - (filterWidth/2)) > -1 && (px + j - (filterWidth/2)) < cols && (px + j - (filterWidth/2)) > -1){
          //      local_pixels[i][j] = d_in[(py + i - (filterWidth/2)) * cols + (px + j - (filterWidth/2))];
            //}
        //}
    //}
  //}


  // Loading the image BLOCK in the shared memory.

  if (global_x > -1 && global_x < cols && global_y > -1 && global_y < rows){
    local_pixels[local_y][local_x] = d_in[global_y * cols + global_x];  
  }
  else
    local_pixels[local_y][local_x] = 0;
  

  // Loading the filter in the shared memory.
  
  if (threadIdx.x < filterWidth && threadIdx.y < filterWidth){
    local_filter[threadIdx.y][threadIdx.x] = d_filter[threadIdx.y * filterWidth + threadIdx.x];
  }

  __syncthreads();

  if (local_x < TILE_WIDTH && local_y < TILE_WIDTH && px < cols && py < rows) {
    float pixval = 0.0;
    for(int blurRow = -(filterWidth / 2); blurRow < (filterWidth / 2) + 1; ++blurRow) {
        for(int blurCol = -(filterWidth / 2); blurCol < (filterWidth / 2) + 1; ++blurCol) {        
            int curRow = local_y + blurRow + filterWidth / 2;
            int curCol = local_x + blurCol + filterWidth / 2;
            pixval += (float) local_pixels[curRow][curCol] * local_filter[(blurRow + (filterWidth/2))][(blurCol + filterWidth/2)];
            //pixval += ((float) d_in[curRow * cols + curCol] * d_filter[(blurRow + (filterWidth/2)) * filterWidth + (blurCol + filterWidth/2)]);
        }
    }
    __syncthreads();
    d_out[py * cols + px] = (unsigned char) pixval;
    //d_out[py * cols + px] = (unsigned char) local_pixels[local_y][local_x];
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
        float *d_filter,  int filterWidth, unsigned char *d_temp){
 


        // Block setup for global and shared memory implementation
        //dim3 blockSize(BLOCK, BLOCK, 1);

        // Grid setup for global memory implementation.
        //dim3 gridSize((cols-1)/BLOCK + 1, (rows-1)/BLOCK + 1, 1);
        
        // Grid setup for shared memory implementation.
        //dim3 gridSize((cols-1)/TILE_WIDTH + 1, (rows-1)/TILE_WIDTH + 1, 1);

        // Block setup for global memory separable kernel row.
        dim3 blockSize(BLOCK, BLOCK, 1);

        // Grid setup for global memory separable kernel row.
        dim3 gridSize((cols-1)/TILE_WIDTH + 1, (rows-1)/TILE_WIDTH + 1, 1);

        dim3 blockSize_col(BLOCKCOL, BLOCKCOL, 1);
        dim3 gridSize_col((cols-1)/BLOCKCOL + 1, (rows-1)/BLOCKCOL + 1, 1);

        separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur_separable_row<<<gridSize, blockSize>>>(d_red, d_temp, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        gaussianBlur_separable_col<<<gridSize_col, blockSize_col>>>(d_temp, d_rblurred, rows, cols, d_filter, filterWidth);
        //gaussianBlur_shared<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter, filterWidth);
        //gaussianBlur<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur_separable_row<<<gridSize, blockSize>>>(d_green, d_temp, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        gaussianBlur_separable_col<<<gridSize_col, blockSize_col>>>(d_temp, d_gblurred, rows, cols, d_filter, filterWidth);
        //gaussianBlur_shared<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter, filterWidth);
        //gaussianBlur<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        gaussianBlur_separable_row<<<gridSize, blockSize>>>(d_blue, d_temp, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        gaussianBlur_separable_col<<<gridSize_col, blockSize_col>>>(d_temp, d_bblurred, rows, cols, d_filter, filterWidth);
        //gaussianBlur_shared<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter, filterWidth);
        //gaussianBlur<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter, filterWidth);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, rows, cols);

        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());   

}





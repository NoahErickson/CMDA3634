/*THIS IS SHELL CODE
compile with nvcc -Xcompiler -fopenmp -o cudaGOLShell cudaGOLShell.cu  -arch=sm_20
run with ./cudaGOLShell boardfile.txt
NOTE: fixed printing for non-square boards.
*/
// libraries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda.h>
/*
Put #define statements here 
*/
#define Treduction 256

// function prototypes
void printBoard(int, int, int* );
void updateBoard(int, int, int*, int*);

/*Q1: put your cudaUpdateBoard kernel here*/
__global__ void cudaUpdateBoard(int N, int M,  int* oldBoard, int* newBoard){
  
  
  // Declare variables
  int i, j;
  int cell, cellBelow, cellAbove;
  int sum, sumBelow, sumLevel, sumAbove;
  int oldState, newState;
  
  for( i=1;i<N+1;++i){  //starting at 1 to skip boundary layer
    for( j=1;j<M+1;++j){ //starting at 1 to skip boundary layer
      
      // Make useful indices
      cell = j+i*(M+2); // Current cell
      cellBelow = j+(i-1)*(M+2); // Cell below it
      cellAbove = j+(i+1)*(M+2); // Cell above it
      
      // Split the sum into the 3 above, 3 below, and 2 level neighbors
      sumBelow = oldBoard[cellBelow]+oldBoard[cellBelow-1]+oldBoard[cellBelow+1];
      sumLevel = oldBoard[cell-1]+oldBoard[cell+1];
      sumAbove = oldBoard[cellAbove]+oldBoard[cellAbove-1]+oldBoard[cellAbove+1];
      
      // Compute the sum
      sum = sumBelow + sumLevel + sumAbove;
      
      
      // Get the current state of the cell
      oldState = oldBoard[cell];
      
      // Game of life rules:
      // If the cell was alive
      if(oldState == 1){
        
        // Exactly 2 or 3 neighbors
        if(sum == 2||sum == 3){
          newState = 1;
        }
        
        // More then 3 or less then 2 neighbors
        else{
          newState = 0;
        }
      }
      
      // If the cell was dead
      else{
        
        // Exactly 3 neighbors
        if(sum == 3){
          newState = 1;
        }
        
        else{
          newState = 0;
        }
      }
      
      // Update new board
      newBoard[cell] = newState;
      
      //
    }
  } 
}


/*Q2: put your cudaUpdateChecker kernel here*/
__global__ void cudaUpdateChecker(int N, int M, int* newBoard, int* oldBoard, int* partialSums){
  int t = threadIdx.x;
  int b = blockIdx.x;
  int id = t + b*Treduction;
  __shared__ int sumBlock[Treduction];
  sumBlock[t] = 0;
  __syncthreads();
  if (id < M*N && id >= 0){
     if (newBoard[id] - oldBoard[id] != 0)
     	sumBlock[t] = 1;
  }
    int alive = Treduction/2;
    while(alive >= 1){
    		__syncthreads();
		if (t < alive) sumBlock[t] += sumBlock[t+alive];
		alive /= 2;
    }
    partialSums[b] = sumBlock[0];
}



// main
int main(int argc, char **argv){
  
  // Board dimensions
  int N, M;
  
  // Read input file containing board information and number of iterations
  FILE *fp = fopen(argv[1], "r");
  
  if(fp==NULL){
    printf("Game Of Life: could not load input file %s\n", argv[1]);
    exit(0);
  }
  // keep reading the file until you find $Size
  char buf[BUFSIZ];
  do{
    fgets(buf, BUFSIZ, fp);
  }while(!strstr(buf, "$Size"));
  
  // read the size
  fgets(buf, BUFSIZ, fp);
  sscanf(buf, "%d %d", &N, &M);
  
  // Initialize boards
  int *boardA = (int*) calloc((N+2)*(M+2), sizeof(int));
  int *boardB = (int*) calloc((N+2)*(M+2), sizeof(int));
  int ii;
  /* DO NOT REMOVE THIS PART */
  for (ii=0; ii<(N+2)*(M+2); ii++){
    boardA[ii] = 0;
    boardB[ii] = 0;
  }
  
  // Read number of updates
  int T;
  do{
    fgets(buf, BUFSIZ, fp);
  }while(!strstr(buf, "$Updates"));
  fgets(buf, BUFSIZ, fp);
  sscanf(buf, "%d", &T);
  
  printf("number of updates: %d\n", T);
  int numAlive;
  
  // next, scan for how many alive cells you have
  do{
    fgets(buf, BUFSIZ, fp);
  }while(!strstr(buf, "$Alive"));
  
  // read the number of alive cells
  fgets(buf, BUFSIZ, fp);
  sscanf(buf, "%d", &numAlive);
  printf("initial number of alive cells:  %d \n", numAlive);
  
  //allocate the alive list (one list per every dimension
  int * LiveList_i = (int*) calloc(numAlive, sizeof(int));
  int * LiveList_j = (int*) calloc(numAlive, sizeof(int));
  
  for (int i=0; i<numAlive; i++){
    fgets(buf, BUFSIZ, fp);
    sscanf(buf, "%d %d", &LiveList_i[i], &LiveList_j[i]);
  }
  
  fclose(fp);
  // Spawn Cells
  for(int n = 0; n<numAlive; ++n){
    int i = LiveList_i[n]; int j = LiveList_j[n];
    boardA[j+i*(M+2)] = 1;
  }
  free(LiveList_i);
  free(LiveList_j);
  
  // Print Initial Board
  printf("Initial Condition\n");
  if ((N<=60) &&(M<=60)){
    printBoard(N,M,boardA);
  }

    
  // Start Game
  int K =M*N;
  /*Q1: create and allocate DEVICE boards A and B here */
  int *DboardA, *DboardB; 
  cudaMalloc(&DboardA, K*sizeof(int));
  cudaMalloc(&DboardB, K*sizeof(int));
  /*Q1: copy boardA to DEVICE boards A here */
  cudaMemcpy(DboardA, boardA, K*sizeof(int), cudaMemcpyHostToDevice);
  
  
  int t = 0;
  int changes;
 
 /*Q1: set the number of blocks and threads here*/
  dim3 grid(4,4);
  dim3 block(8,8);

  /*Q2: set the number of blocks and threads here*/
  int numblocks = K/Treduction;
  int numthreads = Treduction;
  
  /*Q2: create and allocate DEVICE boards for partial sums */
  int *Dsum;
  cudaMalloc(&Dsum, numblocks*sizeof(int));

  /*Q2: create and allocate HOST board for partial sums */
  int *Hsum = (int*) calloc(numblocks, sizeof(int));
 
  // we time using OpenMP timing functions
  double t1, t2;
// REMEMBER TO TURN OFF ALL DISPLAY COMMANDS BEFORE TIMING!!! 
t1= omp_get_wtime();

  while(t<T){
    
    // Update boardA into boardB
    /* Q1: replace this call with cudaUpdateBoard*/
    cudaUpdateBoard <<< grid, block >>> (N, M, DboardA, DboardB);
 
    // check for changes
    /* Q2: call cudaUpdateChecker here*/
    cudaUpdateChecker <<< numblocks, numthreads >>> (N, M, DboardB, DboardA, Dsum);

   /* Q2: copy the partial DEVICE sum array to host here*/
    cudaMemcpy(Hsum,Dsum,numblocks*sizeof(int), cudaMemcpyDeviceToHost);

    changes  = 0;
   /* Q2: sum the entries of the partial sum array, check for if still life*/
    for (int i = 0;i<numblocks;++i){
    	changes += Hsum[i];
}

    cudaMemcpy(boardB,DboardB,K*sizeof(int), cudaMemcpyDeviceToHost);
    if (changes == 0){
       printf("Board has reached still-life\n");
       break;
     }

    //copy for display
   /* Q2: copy DEVICE board A to HOST boardA for display*/
    cudaMemcpy(boardA, DboardA, K*sizeof(int), cudaMemcpyDeviceToHost);

    // display if the board is small
    ++t;
    printf("updated, t = %d\n", t);
    if ((N<=60) &&(M<=60)){
       printBoard(N,M,boardB);
    }
    if(t==T) break;
   
    // Update boardB into boardA
    /* Q1: replace this call with cudaUpdateBoard*/
    cudaUpdateBoard <<< grid, block >>> (N, M, DboardB, DboardA);

    //check for changes
   /* Q2: call cudaUpdateChecker here*/
   cudaUpdateChecker <<< numblocks, numthreads >>> (N, M, DboardA, DboardB, Dsum);
 
   /* Q2: copy the partial DEVICE sum array to host here*/
   cudaMemcpy(Hsum, Dsum, numblocks*sizeof(int), cudaMemcpyDeviceToHost);
    
  /* Q2: sum the entries of the partial sum array, check for if still life*/
    changes  = 0;
    for (int i = 0;i<numblocks;++i)
    	changes += Hsum[i];

    cudaMemcpy(boardA, DboardA, K*sizeof(int), cudaMemcpyDeviceToHost);
    if (changes == 0){
       printf("Board has reached still-life\n");
       break;
    }
  
    //copy board for display
    /* Q2: copy DEVICE board A to HOST boardA for display*/

    //display board if small enough
    ++t;
    printf("updated, t = %d \n", t);
    if ((N<=60) &&(M<=60)){
      printBoard(N,M,boardA);
    }

    if(t==T) break;
    //check for still-life
    
  }
 t2 = omp_get_wtime();
printf("it took %f seconds\n", t2-t1);  
// Finish
  free(boardA);
  free(boardB);
  
  //free cuda variables
  /*Q1 and Q2: free DEVICE variables using cudaFree*/
  cudaFree(DboardA);
  cudaFree(DboardB);
  cudaFree(Dsum);
  cudaFree(Hsum);

  return(0);
}

void updateBoard(int N, int M,  int* oldBoard, int* newBoard){
  
  // Declare variables
  int i, j;
  int cell, cellBelow, cellAbove;
  int sum, sumBelow, sumLevel, sumAbove;
  int oldState, newState;
  
  for( i=1;i<N+1;++i){  //starting at 1 to skip boundary layer
    for( j=1;j<M+1;++j){ //starting at 1 to skip boundary layer
      
      // Make useful indices
      cell = j+i*(M+2); // Current cell
      cellBelow = j+(i-1)*(M+2); // Cell below it
      cellAbove = j+(i+1)*(M+2); // Cell above it
      
      // Split the sum into the 3 above, 3 below, and 2 level neighbors
      sumBelow = oldBoard[cellBelow]+oldBoard[cellBelow-1]+oldBoard[cellBelow+1];
      sumLevel = oldBoard[cell-1]+oldBoard[cell+1];
      sumAbove = oldBoard[cellAbove]+oldBoard[cellAbove-1]+oldBoard[cellAbove+1];
      
      // Compute the sum
      sum = sumBelow + sumLevel + sumAbove;
      
      
      // Get the current state of the cell
      oldState = oldBoard[cell];
      
      // Game of life rules:
      // If the cell was alive
      if(oldState == 1){
        
        // Exactly 2 or 3 neighbors
        if(sum == 2||sum == 3){
          newState = 1;
        }
        
        // More then 3 or less then 2 neighbors
        else{
          newState = 0;
        }
      }
      
      // If the cell was dead
      else{
        
        // Exactly 3 neighbors
        if(sum == 3){
          newState = 1;
        }
        
        else{
          newState = 0;
        }
      }
      
      // Update new board
      newBoard[cell] = newState;
      
      //
    }
  }
  
}


void printBoard(int N, int M, int* board){
  
  int i, j, cell, state;
  //Formatted to start in top left corner, moving across each row
  for(i=1;i<N+1;++i){
    for(j=1;j<M+1;++j){ //starting at 1 to skip boundary layer
      
      // Cell number and state
    
      cell = j + i*(M+2);
   //   printf("i= %d j = %d this is cell %d \n",i,j, cell );
      state = board[cell];
      
      if(state == 1){
        printf("X ");
      }
      
      else{
        printf(". ");
      }
      
    }
    printf("\n");
  }
}

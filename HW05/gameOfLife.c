// HW05 reference shell, CMDA 3634, 2017 Spring

// libraries
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// function prototypes
void printBoard(int, int, int* );
void updateBoard(int, int, int*, int*);
int mainChanges;

// main
int main(int argc, char **argv){
  
  double startTime = omp_get_wtime();
  // ATTENTION!!
  // compile with: gcc -o <RUNNAME> <FILENAME>.c -fopenmp
  // run: ./gameOfLife inputFile.txt
  
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
  int i,n;

  for (i=0; i<numAlive; i++){
    fgets(buf, BUFSIZ, fp);
    sscanf(buf, "%d %d", &LiveList_i[i], &LiveList_j[i]);
  }
  
  fclose(fp);
  // Spawn Cells
  for(n = 0; n<numAlive; ++n){
    int i = LiveList_i[n]; int j = LiveList_j[n];
    boardA[j+i*(N+2)] = 1;
  }
  free(LiveList_i);
  free(LiveList_j);
  
  // Print Initial Board
  printf("Initial Condition\n");
  printBoard(N,M,boardA);
  
  // changeChecker states number of cells with changed state each iteration
  int changeChecker = 0;
  
  // Start Game
  int t = 0;
  while(t<T){
    
    // Update boardA into boardB
    ++t;
    mainChanges = 0;
    updateBoard(N,M,boardA,boardB);
    // Question 2(c). part 1, starts here
    if (mainChanges == 0){
      printf("Iterations stopped, you have reached a still-life board\n");
      break;
    }
    // Question 2(c), part 1, ends here
    
    
    // Display board
    printf("After %d iterations\n",t);
    printBoard(N,M,boardB);
    
    // Check if we're done
    if(t==T) break;
    
    // Update boardB into boardA
    ++t;
    mainChanges = 0;
    updateBoard(N,M,boardB,boardA);
   
    // Question 2(c). part 2, starts here
    if (mainChanges == 0){
      printf("Iterations stopped, you have reached a still-life board\n");
      break;
    }
    // Question 2(c), part 2, ends here
    
    printf("After %d iterations\n",t);
    printBoard(N,M,boardA);
  }
  
  // Finish
  free(boardA);
  free(boardB);
  printf("Elapsed Time: %f\n", omp_get_wtime()-startTime);
  return(0);
}

void updateBoard(int N, int M, int* oldBoard, int* newBoard){

  // Declare variables
  int i, j;
  int cell, cellBelow, cellAbove;
  int sum, sumBelow, sumLevel, sumAbove;
  int oldState, newState;  
  
  // Question 2 (a): declare summation variables here
  int changes;
  int threadChanges;
  // Question 2 (a): ends here
  omp_set_num_threads(16);
  // Question 1 (a) starts here
#pragma omp parallel private(i, j, cell, cellBelow, cellAbove, sum, sumBelow, sumLevel, sumAbove, oldState, newState, threadChanges)
  // Question 1 (a) ends here
  
  // Question 1 (a): place opening bracket { here
  { 
  // Question 1 (d) starts here, part of question 2 (a) starts here too
    changes = 0;
    threadChanges = 0;
  // Question 1 (d) ends here, part of question 2 (a) ends here too
#pragma omp for reduction(+:changes)
    for( i=1;i<M+1;++i){  //starting at 1 to skip boundary layer
      for( j=1;j<N+1;++j){ //starting at 1 to skip boundary layer
	
	// Make useful indices
	cell = j+i*(N+2); // Current cell
	cellBelow = j+(i-1)*(N+2); // Cell below it
	cellAbove = j+(i+1)*(N+2); // Cell above it
      
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
	    changes++;
	    mainChanges++;
	    threadChanges++;
	  }
	}
      
	// If the cell was dead
	else{
        
	  // Exactly 3 neighbors
	  if(sum == 3){
          newState = 1;
	  changes++;
	  mainChanges++;
	  threadChanges++;
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
  // Part of Question 2 (b) starts here
    printf("Thread Index: %d, changes in thread: %d\n", omp_get_thread_num(), threadChanges);
  // Part of Question 2 (b) ends here
  
  // Question 1 (a): place closing bracket } here
  }
  // Part of Question 2 (b) starts here
  printf("total changes: %d\n", changes);
  // Part of Question 2 (b) ends here
  
}


void printBoard(int N, int M, int* board){
  
  int i, j, cell, state;
  //Formatted to start in top left corner, moving across each row
  for(i=1;i<M+1;++i){
    for(j=1;j<N+1;++j){ //starting at 1 to skip boundary layer
      
      // Cell number and state
      cell = j + i*(N+2);
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

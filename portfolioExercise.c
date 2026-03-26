//
// Starting code for the portfolio exercise. Some required routines are included in a separate
// file (ending '_extra.h'); this file should not be altered, as it will be replaced with a different
// version for assessment.
//
// Compile as normal, e.g.,
//
// > gcc -o portfolioExercise portfolioExercise.c
//
// and launch with the problem size N and number of threads p as command line arguments, e.g.,
//
// > ./portfolioExercise 12 4
//


//
// Includes.
//
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#include "portfolioExercise_extra.h"        // Contains routines not essential to the assessment.

//data struct for each thread
typedef struct {
    int thread_id;
    float **M;
    float *u;
    float *v;
    int N;
    int nThreads;
    float *dotprod;
    pthread_mutex_t *mutex;
} thread_data_t;

// parallel calc
void* parrallelcalc(void* arg) {
    thread_data_t* data = (thread_data_t*)arg; 
    //range of rows for thread
    int rows_per_thread = data->N / data->nThreads;
    int start_row = data->thread_id * rows_per_thread;
    int end_row = start_row + rows_per_thread;


    // matrix vector multi
    for (int i = start_row; i < end_row; i++) {
        data->v[i] = 0.0f;
        for (int j = 0; j < data->N; j++) {
            data->v[i] += data->M[i][j] * data->u[j];
        }
    }

    // dot prod calc
    float local_dot = 0.0f;
    for (int i = start_row; i < end_row; i++) {
        local_dot += data->v[i] * data->v[i];
    }

    // add local sum to the global dotprod using the mutex
    pthread_mutex_lock(data->mutex);
    *(data->dotprod) += local_dot;
    pthread_mutex_unlock(data->mutex);
    //sees what thread is doing what calc
    printf("Thread %d handles rows [%d, %d]\n", data->thread_id, start_row, end_row);

    return NULL;
}

//
// Main.
//
int main( int argc, char **argv )
{
    //
    // Initialisation and set-up.
    //

    // Get problem size and number of threads from command line arguments.
    int N, nThreads;
    if( parseCmdLineArgs(argc,argv,&N,&nThreads)==-1 ) return EXIT_FAILURE;

    // Initialise (i.e, allocate memory and assign values to) the matrix and the vectors.
    float **M, *u, *v;
    if( initialiseMatrixAndVector(N,&M,&u,&v)==-1 ) return EXIT_FAILURE;

    // For debugging purposes; only display small problems (e.g., N=8 and nThreads=2 or 4).
    if( N<=12 ) displayProblem( N, M, u, v );

    // Start the timing now.
    struct timespec startTime, endTime;
    clock_gettime( CLOCK_REALTIME, &startTime );

    pthread_mutex_t dot_mutex;  //mutex for dot prod reduction
    pthread_mutex_init(&dot_mutex, NULL);


    //
    // Parallel operations, timed.
    //
    float dotprod = 0.0f;        // You should leave the result of your calculation in this variable.

    pthread_t *threads = malloc(nThreads * sizeof(pthread_t));
    thread_data_t *thread_args = malloc(nThreads * sizeof(thread_data_t));

    // Step 1. Matrix-vector multiplication Mu = v.
    // Step 2. The dot product of the vector v with itself.
    // Both steps are handled within the thread function to maximize parallelism.
    for (int i = 0; i < nThreads; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].M = M;
        thread_args[i].u = u;
        thread_args[i].v = v;
        thread_args[i].N = N;
        thread_args[i].nThreads = nThreads;
        thread_args[i].dotprod = &dotprod;
        thread_args[i].mutex = &dot_mutex;
        pthread_create(&threads[i], NULL, parrallelcalc, &thread_args[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Clean up threading resources
    pthread_mutex_destroy(&dot_mutex);
    free(threads);
    free(thread_args);

    // After completing Step 1, you can uncomment the following line to display M, u and v, to check your solution so far.
    if( N<=12 ) displayProblem( N, M, u, v );

    // DO NOT REMOVE OR MODIFY THIS PRINT STATEMENT AS IT IS REQUIRED BY THE ASSESSMENT.
    printf( "Result of parallel calculation: %f\n", dotprod );

    //
    // Check against the serial calculation.
    //

    // Output final time taken.
    clock_gettime( CLOCK_REALTIME, &endTime );
    double seconds = (double)( endTime.tv_sec + 1e-9*endTime.tv_nsec - startTime.tv_sec - 1e-9*startTime.tv_nsec );
    printf( "Time for parallel calculations: %g secs.\n", seconds );

    // Step 1. Matrix-vector multiplication Mu = v.
    for( int row=0; row<N; row++ )
    {
        v[row] = 0.0f;              // Make sure the right-hand side vector is initially zero.

        for( int col=0; col<N; col++ )
            v[row] += M[row][col] * u[col];
    }

    // Step 2: The dot product of the vector v with itself
    float dotProduct_serial = 0.0f;
    for( int i=0; i<N; i++ ) dotProduct_serial += v[i]*v[i];

    // DO NOT REMOVE OR MODIFY THIS PRINT STATEMENT AS IT IS REQUIRED BY THE ASSESSMENT.
    printf( "Result of the serial calculation: %f\n", dotProduct_serial );

    //
    // Clear up and quit.
    //
    freeMatrixAndVector( N, M, u, v );

    return EXIT_SUCCESS;
}
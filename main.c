#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mnist_helper.h"
#include "neural_network.h"
#include "optimiser.h"

void print_help_and_exit(char **argv) {
    printf("usage: %s <path_to_dataset> <learning_rate> <batch_size> <total_epochs>\n", argv[0]);
    exit(0);
}

int main(int argc, char** argv) {
    
    if(argc != 5){
        printf("ERROR: incorrect number of arguments\n");
        print_help_and_exit(argv);
    }
    
    const char* path_to_dataset = argv[1];
    double learning_rate = atof(argv[2]);
    unsigned int batch_size = atoi(argv[3]);
    unsigned int total_epochs = atoi(argv[4]);
    
    if(!path_to_dataset || !learning_rate || !batch_size || !total_epochs) {
        printf("ERROR: invalid argument\n");
        print_help_and_exit(argv);
    }
    
    printf("********************************************************************************\n");
    printf("Initialising Dataset... \n");
    printf("********************************************************************************\n");
    initialise_dataset(path_to_dataset,
                       0 // print flag
                       );

    printf("********************************************************************************\n");
    printf("Initialising neural network... \n");
    printf("********************************************************************************\n");
    initialise_nn();

    printf("********************************************************************************\n");
    printf("Initialising optimiser...\n");
    printf("********************************************************************************\n");
    initialise_optimiser(learning_rate, batch_size, total_epochs);

    printf("********************************************************************************\n");
    printf("Performing training optimisation...\n");
    printf("********************************************************************************\n");
    run_optimisation();
    
    printf("********************************************************************************\n");
    printf("Program complete... \n");
    printf("********************************************************************************\n");
    free_dataset_data_structures();
    return 0;
}

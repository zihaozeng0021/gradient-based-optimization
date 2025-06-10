#include "mnist_helper.h"
#include "stdio.h"
#include <stdlib.h>
#include <string.h>

#define TRAINING_DATA_BYTES  47040016 // 4 + 4 + 4 + 4 + (28*28)*60000
#define TRAINING_LABELS_BYTES  60008 // 4 + 4 + 60000
#define TESTING_DATA_BYTES  7840016 // 4 + 4 + 4 + 4 + (28*28)*10000
#define TESTING_LABELS_BYTES  10008 // 4 + 4 + 60000

#define PATH_SIZE_MAX 200

uint8_t** training_data;
uint8_t training_labels[N_TRAINING_SET];
uint8_t** testing_data;
uint8_t testing_labels[N_TESTING_SET];

uint8_t training_data_buffer[TRAINING_DATA_BYTES];
uint8_t training_labels_buffer[TRAINING_LABELS_BYTES];
uint8_t testing_data_buffer[TESTING_DATA_BYTES];
uint8_t testing_labels_buffer[TESTING_LABELS_BYTES];

void initialise_dataset(const char* path_to_dataset, unsigned int print_samples){
    training_data = (uint8_t**) malloc(N_TRAINING_SET * sizeof(double*));
    for (int i = 0; i < N_TRAINING_SET; i++){
        training_data[i] = (uint8_t*) malloc(PIXEL_DIM_FLAT * sizeof(uint8_t));
    }
    testing_data = (uint8_t**) malloc(N_TESTING_SET * sizeof(double*));
    for (int i = 0; i < N_TESTING_SET; i++){
        testing_data[i] = (uint8_t*) malloc(PIXEL_DIM_FLAT * sizeof(uint8_t));
    }
    
    char* training_set_loc;
    char* training_labels_loc;
    char* testing_set_loc;
    char* testing_labels_loc;

    training_set_loc = (char *) malloc(PATH_SIZE_MAX * sizeof(char));
    training_labels_loc = (char *) malloc(PATH_SIZE_MAX * sizeof(char));
    testing_set_loc = (char *) malloc(PATH_SIZE_MAX * sizeof(char));
    testing_labels_loc = (char *) malloc(PATH_SIZE_MAX * sizeof(char));
    
    strcat(strcpy(training_set_loc, path_to_dataset), "/train-images-idx3-ubyte");
    strcat(strcpy(training_labels_loc, path_to_dataset), "/train-labels-idx1-ubyte");
    strcat(strcpy(testing_set_loc, path_to_dataset), "/t10k-images-idx3-ubyte");
    strcat(strcpy(testing_labels_loc, path_to_dataset), "/t10k-labels-idx1-ubyte");
    
    load_mnist_training_set(training_set_loc);
    load_mnist_training_labels(training_labels_loc);
    load_mnist_testing_set(testing_set_loc);
    load_mnist_testing_labels(testing_labels_loc);
    
    free(training_set_loc);
    free(training_labels_loc);
    free(testing_set_loc);
    free(testing_labels_loc);
    
    // test training and testing sets
    if (print_samples){
        for (int i=0; i<3; i++){
            printf("label: %u \n", training_labels[i]);
            print_single_example(training_data, i);
        }
        
        for (int i=0; i<3; i++){
            printf("label: %u \n", testing_labels[i]);
            print_single_example(testing_data, i);
        }
    }
}


void load_mnist_training_set(char* path_to_training_set){
    printf("Loading training set...\n");
    FILE *in_file;
    in_file = fopen(path_to_training_set,"rb");
    fread(training_data_buffer, sizeof(training_data_buffer[0]), TRAINING_DATA_BYTES, in_file);
    fclose(in_file);
    
    uint32_t magic_number = (training_data_buffer[0] << 24) + (training_data_buffer[1] << 16) + (training_data_buffer[2] << 8) + (training_data_buffer[3]);
    uint32_t set_size = (training_data_buffer[4] << 24) + (training_data_buffer[5] << 16) + (training_data_buffer[6] << 8) + (training_data_buffer[7]);
    uint32_t x_dim = (training_data_buffer[8] << 24) + (training_data_buffer[9] << 16) + (training_data_buffer[10] << 8) + (training_data_buffer[11]);
    uint32_t y_dim = (training_data_buffer[12] << 24) + (training_data_buffer[13] << 16) + (training_data_buffer[14] << 8) + (training_data_buffer[15]);
    
    printf("Magic number: %u\n", magic_number);
    printf("Set size: %u\n", set_size);
    printf("x_dim: %u\n", x_dim);
    printf("y_dim: %u\n", y_dim);

    for (int i=0; i<N_TRAINING_SET; i++){
        for (int j=0; j<PIXEL_DIM_FLAT; j++){
            training_data[i][j] = training_data_buffer[i*PIXEL_DIM_FLAT + j + 16];
        }
    }
    printf("Training set loaded successfully...\n");
}

void load_mnist_training_labels(char* path_to_training_set_labels){
    printf("\nLoading training set labels...\n");
    FILE *in_file;
    in_file = fopen(path_to_training_set_labels,"rb");
    fread(training_labels_buffer, sizeof(training_labels_buffer[0]), TRAINING_LABELS_BYTES, in_file);
    fclose(in_file);
    
    // Read headers
    uint32_t magic_number = (training_labels_buffer[0] << 24) + (training_labels_buffer[1] << 16) + (training_labels_buffer[2] << 8) + (training_labels_buffer[3]);
    uint32_t set_size = (training_labels_buffer[4] << 24) + (training_labels_buffer[5] << 16) + (training_labels_buffer[6] << 8) + (training_labels_buffer[7]);
    printf("Magic number: %u\n", magic_number);
    printf("Set size: %u\n", set_size);
    
    // Read label data
    for (int i=0; i<N_TRAINING_SET; i++){
        training_labels[i] = training_labels_buffer[i + 8];
    }
    printf("Training set labels loaded successfully...\n");
}

void load_mnist_testing_set(char* path_to_testing_set){
    printf("\nLoading testing set...\n");
    FILE *in_file;
    in_file = fopen(path_to_testing_set,"rb");
    fread(testing_data_buffer, sizeof(testing_data_buffer[0]), TESTING_DATA_BYTES, in_file);
    fclose(in_file);
    
    // Read headers
    uint32_t magic_number = (testing_data_buffer[0] << 24) + (testing_data_buffer[1] << 16) + (testing_data_buffer[2] << 8) + (testing_data_buffer[3]);
    uint32_t set_size = (testing_data_buffer[4] << 24) + (testing_data_buffer[5] << 16) + (testing_data_buffer[6] << 8) + (testing_data_buffer[7]);
    uint32_t x_dim = (testing_data_buffer[8] << 24) + (testing_data_buffer[9] << 16) + (testing_data_buffer[10] << 8) + (testing_data_buffer[11]);
    uint32_t y_dim = (testing_data_buffer[12] << 24) + (testing_data_buffer[13] << 16) + (testing_data_buffer[14] << 8) + (testing_data_buffer[15]);
    printf("Magic number: %u\n", magic_number);
    printf("Set size: %u\n", set_size);
    printf("x_dim: %u\n", x_dim);
    printf("y_dim: %u\n", y_dim);
    
    // Read pixel data
    for (int i=0; i<N_TESTING_SET; i++){
        for (int j=0; j<PIXEL_DIM_FLAT; j++){
            testing_data[i][j] = testing_data_buffer[i*PIXEL_DIM_FLAT + j + 16];
        }
    }
    printf("Testing set loaded successfully...\n");
}

void load_mnist_testing_labels(char* path_to_testing_set_labels){
    
    // Open, read  and close file
    printf("\nLoading testing set labels...\n");
    FILE *in_file;
    in_file = fopen(path_to_testing_set_labels,"rb");
    fread(testing_labels_buffer, sizeof(testing_labels_buffer[0]), TESTING_LABELS_BYTES, in_file);
    fclose(in_file);
    
    // Read headers
    uint32_t magic_number = (testing_labels_buffer[0] << 24) + (testing_labels_buffer[1] << 16) + (testing_labels_buffer[2] << 8) + (testing_labels_buffer[3]);
    uint32_t set_size = (testing_labels_buffer[4] << 24) + (testing_labels_buffer[5] << 16) + (testing_labels_buffer[6] << 8) + (testing_labels_buffer[7]);
    printf("Magic number: %u\n", magic_number);
    printf("Set size: %u\n", set_size);
    
    // Read label data
    for (int i=0; i<N_TESTING_SET; i++){
        testing_labels[i] = testing_labels_buffer[i + 8];
    }
    printf("Testing set labels loaded successfully...\n\n");
}

void free_dataset_data_structures(void){
    for (int i = 0; i < N_TRAINING_SET; i++){
        free(training_data[i]);
    }

    for (int i = 0; i < N_TESTING_SET; i++){
        free(testing_data[i]);
    }
    
    free(training_data);
    free(testing_data);
}

void print_single_example(uint8_t** dataset, int n){
    for (int i=0; i<PIXEL_DIM; i++){
        for (int j=0; j<PIXEL_DIM; j++){
            printf("%3i ", dataset[n][i*PIXEL_DIM + j]);
        }
        printf("\n");
    }
    printf("\n");
}

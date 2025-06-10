#ifndef MNIST_HELPER_H
#define MNIST_HELPER_H

#include <stdint.h>

#define N_TRAINING_SET 60000
#define N_TESTING_SET 10000
#define PIXEL_DIM 28 // MNIST input array is 28 by 28 pixels
#define PIXEL_DIM_FLAT 784 // 784 = 28 * 28

extern uint8_t** training_data;
extern uint8_t training_labels[N_TRAINING_SET];
extern uint8_t** testing_data;
extern uint8_t testing_labels[N_TESTING_SET];

void initialise_dataset(const char* path_to_dataset, unsigned int print_samples);
void load_mnist_training_set(char* words);
void load_mnist_training_labels(char* path_to_training_set);
void load_mnist_testing_set(char* path_to_training_set);
void load_mnist_testing_labels(char* path_to_training_set);
void free_dataset_data_structures(void);
void print_single_example(uint8_t** dataset, int n);
void print_string(char* words);


#endif /* MNIST_HELPER_H */



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/times.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>
#include "optimiser.h"
#include "mnist_helper.h"
#include "neural_network.h"

unsigned int log_freq = 30000;
unsigned int num_batches;
unsigned int batch_size;
double learning_rate;
unsigned int total_epochs;
unsigned int forward_differencing = 0;
unsigned int backward_differencing = 0;
unsigned int central_differencing = 0;
unsigned int learning_rate_decay = 0;
unsigned int momentum = 1;
double initial_learning_rate;
#define FINAL_LEARNING_RATE 0.0001
#define MOMENTUM_COEF 0.1
#define ADAM_BETA1 0.9
#define ADAM_BETA2 0.999
#define ADAM_EPSILON 1e-8
unsigned int adam_iter = 1;

void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy) {
    printf("Epoch: %u, Total iter: %u, Mean Loss: %0.12f, Test Acc: %f\n", epoch_counter, total_iter, mean_loss, test_accuracy);
}

void initialise_optimiser(double cmd_line_learning_rate, int cmd_line_batch_size, int cmd_line_total_epochs) {
    batch_size = cmd_line_batch_size;
    learning_rate = cmd_line_learning_rate;
    initial_learning_rate = cmd_line_learning_rate;
    total_epochs = cmd_line_total_epochs;
    num_batches = total_epochs * (N_TRAINING_SET / batch_size);
    printf("Optimising with parameters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = %u\n\tlearning_rate = %f\n\n", total_epochs, batch_size, num_batches, learning_rate);
}

void free_2d_array(double **array, int rows) {
    if (array == NULL)
        return;
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

void ensure_directory_exists(const char* dirname) {
    struct stat st = {0};
    if (stat(dirname, &st) == -1) {
        if (mkdir(dirname, 0755) != 0) {
            perror("Error creating directory");
            exit(EXIT_FAILURE);
        }
    }
}

void get_log_filenames(char* opt_filename, char* valid_filename) {
    sprintf(opt_filename, "data/part3/training_stats_00001_1000.dat");
    sprintf(valid_filename, "data/part2/validation_stats.dat");
}

FILE* open_file(const char* filename, const char* mode) {
    FILE* f = fopen(filename, mode);
    if (f == NULL) {
        perror("Error opening log file");
        exit(EXIT_FAILURE);
    }
    return f;
}

void log_stats_if_needed(unsigned int total_iter, unsigned int log_freq, double* mean_loss, unsigned int epoch_counter, FILE* train_f) {
    if (total_iter % log_freq == 0 || total_iter == 0) {
        if (total_iter > 0) {
            *mean_loss /= log_freq;
        }
        double test_accuracy = evaluate_testing_accuracy();
        print_training_stats(epoch_counter, total_iter, *mean_loss, test_accuracy);
        fprintf(train_f, "%d %u %f %f\n", epoch_counter, total_iter, *mean_loss, test_accuracy);
        *mean_loss = 0.0;
    }
}

void validate_gradients_if_needed(unsigned int total_iter, unsigned int batch_size, unsigned int sample, FILE* valid_f) {
    if (total_iter == batch_size - 1 && valid_f != NULL) {
        validate_gradients(sample, valid_f);
    }
}

double validate_gradients(unsigned int sample, FILE* f) {
    double epsilon = 1e-8;
    double diff_accumulated = 0.0;
    double rel_diff_accumulated = 0.0;
    double time_spent, avg_diff = 0, avg_rel_diff;
    clock_t start, end;
    int rel_iter = 0;
    if (forward_differencing) {
        start = clock();
        for (int i = 0; i < N_NEURONS_L3; i++) {
            for (int j = 0; j < N_NEURONS_LO; j++) {
                w_L3_LO[i][j].w += epsilon;
                evaluate_forward_pass(training_data, sample);
                double perturbed_loss_plus_eps = compute_xent_loss(training_labels[sample]);
                w_L3_LO[i][j].w -= epsilon;
                evaluate_forward_pass(training_data, sample);
                double perturbed_loss = compute_xent_loss(training_labels[sample]);
                double numerical_grad = (perturbed_loss_plus_eps - perturbed_loss) / epsilon;
                double analytical_grad = dL_dW_L3_LO[0][i + (N_NEURONS_L3 * j)];
                double diff = fabs(numerical_grad - analytical_grad);
                double rel_diff = (diff / fabs(analytical_grad)) * 100.0;
                diff_accumulated += diff;
                if (analytical_grad > 0.0) {
                    rel_iter++;
                    rel_diff_accumulated += rel_diff;
                    fprintf(f, "%f ", rel_diff);
                }
            }
        }
        fprintf(f, "\n");
        end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        avg_diff = diff_accumulated / (N_NEURONS_L3 * N_NEURONS_LO);
        avg_rel_diff = rel_diff_accumulated / rel_iter;
        printf("Forward Diff: Average diff: %.32f, Percentage avg rel_diff: %.32f, Time: %.5f\n", avg_diff, avg_rel_diff, time_spent);
    }
    if (backward_differencing) {
        diff_accumulated = 0.0;
        rel_diff_accumulated = 0.0;
        rel_iter = 0;
        start = clock();
        for (int i = 0; i < N_NEURONS_L3; i++) {
            for (int j = 0; j < N_NEURONS_LO; j++) {
                w_L3_LO[i][j].w -= epsilon;
                evaluate_forward_pass(training_data, sample);
                double perturbed_loss_minus_eps = compute_xent_loss(training_labels[sample]);
                w_L3_LO[i][j].w += epsilon;
                evaluate_forward_pass(training_data, sample);
                double perturbed_loss = compute_xent_loss(training_labels[sample]);
                double numerical_grad = (perturbed_loss - perturbed_loss_minus_eps) / epsilon;
                double analytical_grad = dL_dW_L3_LO[0][i + (N_NEURONS_L3 * j)];
                double diff = fabs(numerical_grad - analytical_grad);
                double rel_diff = (diff / fabs(analytical_grad)) * 100.0;
                diff_accumulated += diff;
                if (analytical_grad > 0.0) {
                    rel_iter++;
                    rel_diff_accumulated += rel_diff;
                    fprintf(f, "%f ", rel_diff);
                }
            }
        }
        fprintf(f, "\n");
        end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        avg_diff = diff_accumulated / (N_NEURONS_L3 * N_NEURONS_LO);
        avg_rel_diff = rel_diff_accumulated / rel_iter;
        printf("Backward Diff: Average diff: %.32f, Percentage avg rel_diff: %.32f, Time: %.5f\n", avg_diff, avg_rel_diff, time_spent);
    }
    if (central_differencing) {
        diff_accumulated = 0.0;
        rel_diff_accumulated = 0.0;
        rel_iter = 0;
        start = clock();
        for (int i = 0; i < N_NEURONS_L3; i++) {
            for (int j = 0; j < N_NEURONS_LO; j++) {
                w_L3_LO[i][j].w += epsilon;
                evaluate_forward_pass(training_data, sample);
                double perturbed_loss_plus_eps = compute_xent_loss(training_labels[sample]);
                w_L3_LO[i][j].w -= 2 * epsilon;
                evaluate_forward_pass(training_data, sample);
                double perturbed_loss_minus_eps = compute_xent_loss(training_labels[sample]);
                w_L3_LO[i][j].w += epsilon;
                double numerical_grad = (perturbed_loss_plus_eps - perturbed_loss_minus_eps) / (2 * epsilon);
                double analytical_grad = dL_dW_L3_LO[0][i + (N_NEURONS_L3 * j)];
                double diff = fabs(numerical_grad - analytical_grad);
                double rel_diff = (diff / fabs(analytical_grad)) * 100.0;
                diff_accumulated += diff;
                if (analytical_grad > 0.0) {
                    rel_iter++;
                    rel_diff_accumulated += rel_diff;
                    fprintf(f, "%f ", rel_diff);
                }
            }
        }
        fprintf(f, "\n");
        end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        avg_diff = diff_accumulated / (N_NEURONS_L3 * N_NEURONS_LO);
        avg_rel_diff = rel_diff_accumulated / rel_iter;
        printf("Central Diff: Average diff: %.32f, Percentage avg rel_diff: %.32f, Time: %.5f\n", avg_diff, avg_rel_diff, time_spent);
    }
    return avg_diff;
}

double evaluate_objective_function(unsigned int sample) {
    evaluate_forward_pass(training_data, sample);
    double loss = compute_xent_loss(training_labels[sample]);
    evaluate_backward_pass_sparse(training_labels[sample], sample);
    store_gradient_contributions();
    return loss;
}

void update_weights_adam(unsigned int N_NEURONS_I, unsigned int N_NEURONS_O, weight_struct_t w_I_O[N_NEURONS_I][N_NEURONS_O], unsigned int batch_size) {
    int i, j;
    for (i = 0; i < N_NEURONS_I; i++) {
        for (j = 0; j < N_NEURONS_O; j++) {
            double grad = w_I_O[i][j].dw / (double) batch_size;
            w_I_O[i][j].m = ADAM_BETA1 * w_I_O[i][j].m + (1.0 - ADAM_BETA1) * grad;
            w_I_O[i][j].v = ADAM_BETA2 * w_I_O[i][j].v + (1.0 - ADAM_BETA2) * (grad * grad);
            double m_hat = w_I_O[i][j].m / (1.0 - pow(ADAM_BETA1, adam_iter));
            double v_hat = w_I_O[i][j].v / (1.0 - pow(ADAM_BETA2, adam_iter));
            w_I_O[i][j].w -= learning_rate * m_hat / (sqrt(v_hat) + ADAM_EPSILON);
            w_I_O[i][j].dw = 0.0;
        }
    }
}

void update_parameters_adam(unsigned int batch_size) {
    adam_iter++;
    update_weights_adam(N_NEURONS_L3, N_NEURONS_LO, w_L3_LO, batch_size);
    update_weights_adam(N_NEURONS_L2, N_NEURONS_L3, w_L2_L3, batch_size);
    update_weights_adam(N_NEURONS_L1, N_NEURONS_L2, w_L1_L2, batch_size);
    update_weights_adam(N_NEURONS_LI, N_NEURONS_L1, w_LI_L1, batch_size);
}

void run_optimisation(void) {
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double mean_loss = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;
    double obj_func = 0.0;
    printf("Using ADAM\n");
    ensure_directory_exists("data");
    ensure_directory_exists("data/part1");
    char optimization_filename[100], validation_filename[100];
    get_log_filenames(optimization_filename, validation_filename);
    FILE *train_f = open_file(optimization_filename, "w");
    FILE *valid_f = NULL;
    if (forward_differencing || backward_differencing || central_differencing) {
        valid_f = open_file(validation_filename, "w");
    }
    clock_t start = clock();
    for (int i = 0; i < num_batches; i++) {
        for (int j = 0; j < batch_size; j++) {
            log_stats_if_needed(total_iter, log_freq, &mean_loss, epoch_counter, train_f);
            obj_func = evaluate_objective_function(training_sample);
            mean_loss += obj_func;
            validate_gradients_if_needed(total_iter, batch_size, training_sample, valid_f);
            total_iter++;
            training_sample++;
            if (training_sample == N_TRAINING_SET) {
                training_sample = 0;
                epoch_counter++;
                if (learning_rate_decay) {
                    double alpha = (double)epoch_counter / (double)total_epochs;
                    learning_rate = initial_learning_rate * (1.0 - alpha) + FINAL_LEARNING_RATE * alpha;
                    printf("Epoch %u: Updated learning rate = %f\n", epoch_counter, learning_rate);
                }
            }
        }
        update_parameters_adam(batch_size);
    }
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time spent: %f\n", time_spent);
    test_accuracy = evaluate_testing_accuracy();
    print_training_stats(epoch_counter, total_iter, (mean_loss / ((double) log_freq)), test_accuracy);
    fclose(train_f);
    if (valid_f != NULL) {
        fclose(valid_f);
    }
}

all: mnist_optimiser.out

mnist_optimiser.out:
	gcc -o mnist_optimiser.out main.c mnist_helper.c neural_network.c optimiser.c -lm -O3

clean:
	rm mnist_optimiser.out

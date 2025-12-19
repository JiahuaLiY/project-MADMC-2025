#ifndef __MODP__
#define __MODP__

#define DEFAULT_VECTOR_SIZE 64

typedef struct {
    int dim;
    int size;
    int maxSize;
    int **ndPoints;
} NDContainer;

NDContainer modp(int **values, int *weights, int capacity, int m, int n);
#endif

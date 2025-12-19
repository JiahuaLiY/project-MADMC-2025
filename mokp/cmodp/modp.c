#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "modp.h"

static NDContainer createNDContainer(int initSize, int dim) {
    NDContainer ndContainer;
    if (!(ndContainer.ndPoints = (int **)malloc(sizeof(int *) * initSize))) {
        fprintf(stderr,
                "Memory error: run out of memory\n");
        exit(EXIT_FAILURE);
    }

    ndContainer.dim = dim;
    ndContainer.size = 0;
    ndContainer.maxSize = initSize;
    for (int i = 0; i < initSize; i++) {
        ndContainer.ndPoints[i] = NULL;
    }
    return ndContainer;
}

static int *createPoint(int dim) {
    int *point;
    if (!(point = (int *)malloc(sizeof(int) * dim))) {
        fprintf(stderr,
                "Memory error: run out of memory\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 0; i++) {
        point[i] = 0;
    }
    return point;
}

static int *clonePoint(int *srcPoint, int dim) {
    int *destPoint;
    if (!(destPoint = (int *)malloc(sizeof(int) * dim))) {
        fprintf(stderr,
                "Memory error: run out of memory\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < dim; i++) {
        destPoint[i] = srcPoint[i];
    }
    return destPoint;
}

static NDContainer cloneNDContainer(NDContainer srcContainer) {
    NDContainer destContainer = createNDContainer(srcContainer.maxSize, srcContainer.dim);

    // Clone all non-dominated points.
    for (int i = 0; i < srcContainer.size; i++) {
        destContainer.ndPoints[i] = clonePoint(srcContainer.ndPoints[i], srcContainer.dim);
    }
    destContainer.size = srcContainer.size;
    return destContainer;
}

static NDContainer add(NDContainer srcContainer, int *vector) {
    NDContainer destContainer = createNDContainer(srcContainer.maxSize, srcContainer.dim);
    
    for (int i = 0; i < srcContainer.size; i++) {
        destContainer.ndPoints[i] = createPoint(srcContainer.dim);
        for (int k = 0; k < srcContainer.dim; k++) {
            destContainer.ndPoints[i][k] = srcContainer.ndPoints[i][k] + vector[k];
        }
    }
    destContainer.size = srcContainer.size;
    return destContainer;
}

static bool equals(int *x, int *y, int dim) {
    for (int i = 0; i < dim; i++) {
        if (x[i] != y[i]) {
            return false;
        }
    }
    return true;
}

static bool paretoDominate(int *x, int *y, int dim) {
    if (equals(x, y, dim)) {
        return false;
    }

    for (int i = 0; i < dim; i++) {
        if (x[i] < y[i]) {
            return false;
        }
    }
    return true;
}

static void insertNDPoint(NDContainer *ndContainer, int *point) {
    if (ndContainer->size >= ndContainer->maxSize) {
        int newMaxSize = ndContainer->maxSize * 2;
        if (!(ndContainer->ndPoints = (int **)realloc(ndContainer->ndPoints, sizeof(int *) * newMaxSize))) {
            fprintf(stderr,
                    "Memory error: run out of memory\n");
            exit(EXIT_FAILURE);
        }

        for (int i = ndContainer->size; i < newMaxSize; i++) {
            ndContainer->ndPoints[i] = NULL;
        }
        ndContainer->maxSize = newMaxSize;
    }

    ndContainer->ndPoints[ndContainer->size] = clonePoint(point, ndContainer->dim);
    ndContainer->size++;
}

static NDContainer mergeNDContainer(NDContainer X, NDContainer Y) {
    NDContainer ndContainer = createNDContainer(DEFAULT_VECTOR_SIZE, X.dim);

    bool isND;
    for (int i = 0; i < X.size; i++) {
        isND = true;

        for (int j = 0; j < Y.size; j++) {
            if (paretoDominate(Y.ndPoints[j], X.ndPoints[i], X.dim)) {
                isND = false;
                break;
            }
        }

        if (isND) {
            insertNDPoint(&ndContainer, X.ndPoints[i]);
        }
    }

    for (int i = 0; i < Y.size; i++) {
        isND = true;

        for (int j = 0; j < X.size; j++) {
            if (equals(X.ndPoints[j], Y.ndPoints[i], X.dim) || paretoDominate(X.ndPoints[j], Y.ndPoints[i], X.dim)) {
                isND = false;
                break;
            }
        }

        if (isND) {
            insertNDPoint(&ndContainer, Y.ndPoints[i]);
        }
    }
    
    return ndContainer;
}

static void cleanNDContainer(NDContainer *ndContainer) {
    for (int i = 0; i < ndContainer->size; i++) {
        if (ndContainer->ndPoints[i] != NULL)
            free(ndContainer->ndPoints[i]);
    }
    free(ndContainer->ndPoints);
    ndContainer->ndPoints = NULL;
    ndContainer->size = 0;
}

NDContainer modp(int **values, int *weights, int capacity, int m, int n) {
    NDContainer *prev, *curr;
    if (!(prev = (NDContainer *)malloc(sizeof(NDContainer) * (capacity + 1)))) {
        fprintf(stderr,
                "Memory error: run out of memory\n");
        exit(EXIT_FAILURE);
    }
    for (int j = 0; j < capacity + 1; j++) {
        prev[j] = createNDContainer(DEFAULT_VECTOR_SIZE, n);
        prev[j].ndPoints[0] = createPoint(n);
        prev[j].size = 1;
    }

    if (!(curr = (NDContainer *)malloc(sizeof(NDContainer) * (capacity + 1)))) {
        fprintf(stderr,
                "Memory error: run out of memory\n");
        exit(EXIT_FAILURE);
    }
    curr[0] = createNDContainer(DEFAULT_VECTOR_SIZE, n);
    curr[0].ndPoints[0] = createPoint(n);
    curr[0].size = 1;

    for (int i = 1; i < m + 1; i++) {
        for (int j = 1; j < capacity + 1; j++) {
            if (j < weights[i - 1]) {
                curr[j] = cloneNDContainer(prev[j]);
            }
            else {
                NDContainer temp = add(prev[j - weights[i - 1]], values[i - 1]);
                curr[j] = mergeNDContainer(temp, prev[j]);
                cleanNDContainer(&temp);
            }
        }
        
        NDContainer *ptr = prev;
        prev = curr;
        curr = ptr;
        for (int j = 1; j < capacity + 1; j++)
            cleanNDContainer(&curr[j]);
    }

    for (int j = 0; j < capacity + 1; j++) {
        cleanNDContainer(&curr[j]);
    }
    free(curr);
    curr = NULL;
    for (int j = 0; j < capacity; j++)
        cleanNDContainer(&prev[j]);
    NDContainer ndContainer = prev[capacity];
    free(prev);
    prev = NULL;

    return ndContainer;
}

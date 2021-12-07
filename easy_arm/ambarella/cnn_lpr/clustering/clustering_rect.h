#ifndef CLUSTERING_RECT_H
#define CLUSTERING_RECT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "image_data_structure.h"

#define myRound(value)		(int)((value) + ( (value) >= 0 ? 0.5 : -0.5))

void clusteringRect(const ResultRect *srcRects, const int srcCount, const float eps, ResultRect *dstRects, int *dstCount);

#ifdef __cplusplus
}
#endif

#endif // CLUSTERING_RECT_H


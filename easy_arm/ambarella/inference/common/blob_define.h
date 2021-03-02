#ifndef BLOBDEFINE_H
#define BLOBDEFINE_H

/**
 * @brief submodule to operate array data
 *
 */
#define BLOB_DATA(a, K, H, W, n, k, h, w) (a)[(( (n) * (K) + (k) ) * (H) + (h) ) * (W) + (w)]
#define DIM2_DATA(a, W, h, w) (a)[(h) * (W) + (w)]
#define DIM1_DATA(a, i) (a)[(i)]

#endif /* BLOBDEFINE_H */
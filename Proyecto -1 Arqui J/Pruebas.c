float sum_array(float* arr, int n);

void compute_stats(
    float* arr,
    int n,
    float* mean,
    float* var,
    float* min,
    float* max
);

void normalize_array(
    float* in,
    float* out,
    int n,
    float mean,
    float stddev
);
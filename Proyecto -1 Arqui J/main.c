#include <stdio.h>
#include <stdlib.h>

extern float sum_array(const float *arr, int n);

void probar(int n)
{
    float *arr = NULL;

    if (n > 0)
    {
        arr = malloc(n * sizeof(float));

        for (int i = 0; i < n; i++)
        {
            arr[i] = (float)(i + 1);
        }
    }

    float resultado = sum_array(arr, n);

    printf("N = %d  |  Suma = %.2f\n", n, resultado);

    free(arr);
}

int main(void)
{
    probar(0);
    probar(1);
    probar(7);
    probar(8);
    probar(15);
    probar(16);
    probar(1000);

    return 0;
}
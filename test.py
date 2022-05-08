static void gemm_v1(float* matA, float* matB, float* matC, const int M, const int N, const int K, const int strideA, const int strideB, const int strideC)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < K; k++)
			{
				sum += matA[j*strideA + k] * matB[k*strideB + i];
			}
			matC[j*strideC + i] = sum;
		}
	}
}

static void gemm_v2(float* matA, float* matB, float* matC, const int M, const int N, const int K, const int strideA, const int strideB, const int strideC)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < K; k++)
			{
				sum += matA[i*strideA + k] * matB[k*strideB + j];
			}
			matC[i*strideC + j] = sum;
		}
	}
}

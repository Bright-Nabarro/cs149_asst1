#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>


__m256 fabs_ps(__m256 x) {
    // 创建一个掩码，将符号位清零
    constexpr int32_t abs_mask_val = 0x7FFFFFFF;
    static const __m256 abs_mask = _mm256_castsi256_ps( _mm256_set1_epi32(abs_mask_val));
    return _mm256_and_ps(x, abs_mask);
}


__m256 fabs_ps(__m256 src, __mmask8 mask, __m256 x)
{
	__m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
	auto ret = _mm256_mask_and_ps(src, mask, x, abs_mask);
	return ret;
}

void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[]);

void sqrt_avx2(int N,
                float initialGuess,
                float values[],
                float output[])
{
	constexpr int fWidth = 8;
	static const float kThreshold = 0.00001f;
	static __m256 vkThreshold = _mm256_broadcast_ss(&kThreshold);

	static const float onef = 1.f;
	static __m256 vonef = _mm256_broadcast_ss(&onef);
	static const float threef = 3.f;
	static __m256 vthreef =_mm256_broadcast_ss(&threef);
	static const float zerofivef = 0.5f;
	static __m256 vzerofivef =_mm256_broadcast_ss(&zerofivef);

	static const int fgt = _CMP_GT_OS;

	int i = 0;
	while(i + fWidth < N)
	{
		__m256 x = _mm256_load_ps(values+i);
		__m256 guess = _mm256_broadcast_ss(&initialGuess);
		__m256 tmp = _mm256_mul_ps(guess, guess);
		tmp = _mm256_mul_ps(tmp, x);
		tmp = _mm256_sub_ps(tmp, vonef);
		__m256 error = fabs_ps(tmp);
		
		while (true)
		{
			// 满足条件的
			__mmask8 while_mask = _mm256_cmp_ps_mask(error, vkThreshold, fgt);
			if (_cvtmask8_u32(while_mask) == 0)
				break;
			__m256 guess_src = guess;
			__m256 error_src = error;
			__m256 guess_xn = _mm256_mask_mul_ps(guess_src, while_mask, x, guess);
			// x * guess
			guess_xn = _mm256_mask_mul_ps(guess_src, while_mask, x, guess);
			// x * guess * guess
			guess_xn = _mm256_mask_mul_ps(guess_src, while_mask, guess_xn, guess);
			// x * guess * guess * guess
			guess_xn = _mm256_mask_mul_ps(guess_src, while_mask, guess_xn, guess);

			// 3.f * guess
			tmp = _mm256_mask_mul_ps(guess_src, while_mask, vthreef, guess);
			// (3.f * guess - x * guess * guess * guess)
			tmp = _mm256_mask_sub_ps(guess_src, while_mask, tmp, guess_xn);
			// guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
			guess = _mm256_mask_mul_ps(guess_src, while_mask, tmp, vzerofivef);

			// guess * guess * x - 1.f
			tmp = _mm256_mask_mul_ps(error_src, while_mask, guess, guess);
			tmp = _mm256_mask_mul_ps(error_src, while_mask, tmp, x);
			tmp = _mm256_mask_sub_ps(error_src, while_mask, tmp, vonef);
			error = fabs_ps(error_src, while_mask, tmp);
		}
		__m256 vecOutput = _mm256_mul_ps(x, guess);
		_mm256_store_ps(output + i, vecOutput);
		i += fWidth;
	}
	sqrtSerial(N-i, initialGuess, values + i, output + i);
}


void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[])
{
	static const float kThreshold = 0.00001f;

    for (int i=0; i<N; i++) {

        float x = values[i];
        float guess = initialGuess;

        float error = fabs(guess * guess * x - 1.f);

        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}


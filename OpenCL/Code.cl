__kernel void convolution_global(__write_only image2d_t dst,
                                 __read_only  image2d_t src,
                                 int coef_matrix_dim,
                                 __constant float *coef_matrix,
                                 float div,
                                 float bias)
{
    const sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE   |
                               CLK_FILTER_NEAREST);

    const int half_matrix_dim = (coef_matrix_dim / 2);
    int2 loc = (int2)(get_global_id(0), get_global_id(1));
    float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int conv_i = -half_matrix_dim; conv_i <= half_matrix_dim; conv_i++) {
        for (int conv_j = -half_matrix_dim; conv_j <= half_matrix_dim; conv_j++) {
            float4 px = read_imagef(src, sampler, loc + (int2)(conv_j, conv_i));
            convPix += px * coef_matrix[(conv_i + half_matrix_dim) * coef_matrix_dim +
                                        (conv_j + half_matrix_dim)];
        }
     }  
     float4 dstPix = convPix * div + bias;
     write_imagef(dst, loc, dstPix);
	 const int2 pos = {get_global_id(0), get_global_id(1)};

	 uint4 sum = read_imageui(src, sampler, pos);

	 write_imageui(dst, (int2)(pos.x, pos.y), sum);
}


__kernel void sobel_global(__write_only image2d_t dst,
                           __read_only  image2d_t src,
                             float div,
                             float bias)
{
    const sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE   |
                               CLK_FILTER_NEAREST);

    int2 loc = (int2)(get_global_id(0), get_global_id(1));

    float4 sum1 = read_imagef(src, sampler, loc + (int2)(-1,-1)) * -1 +
                  read_imagef(src, sampler, loc + (int2)( 0,-1)) * -2 +
                  read_imagef(src, sampler, loc + (int2)( 1,-1)) * -1 +
                  read_imagef(src, sampler, loc + (int2)(-1, 1)) *  1 +
                  read_imagef(src, sampler, loc + (int2)( 0, 1)) *  2 +
                  read_imagef(src, sampler, loc + (int2)( 1, 1)) *  1;

    float4 sum2 = read_imagef(src, sampler, loc + (int2)(-1,-1)) * -1 +
                  read_imagef(src, sampler, loc + (int2)(-1, 0)) * -2 +
                  read_imagef(src, sampler, loc + (int2)(-1, 1)) * -1 +
                  read_imagef(src, sampler, loc + (int2)( 1,-1)) *  1 +
                  read_imagef(src, sampler, loc + (int2)( 1, 0)) *  2 +
                  read_imagef(src, sampler, loc + (int2)( 1, 1)) *  1;

    float4 dstPix = hypot(sum1, sum2) * div + bias;
    write_imagef(dst, loc, dstPix);
}

__kernel void prewitt_global(__write_only image2d_t dst,
                             __read_only  image2d_t src,
                             float div,
                             float bias)
{
    const sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE   |
                               CLK_FILTER_NEAREST);

    int2 loc = (int2)(get_global_id(0), get_global_id(1));

    float4 sum1 = read_imagef(src, sampler, loc + (int2)(-1,-1)) *  1 +
                  read_imagef(src, sampler, loc + (int2)( 0,-1)) *  1 +
                  read_imagef(src, sampler, loc + (int2)( 1,-1)) *  1 +
                  read_imagef(src, sampler, loc + (int2)(-1, 1)) * -1 +
                  read_imagef(src, sampler, loc + (int2)( 0, 1)) * -1 +
                  read_imagef(src, sampler, loc + (int2)( 1, 1)) * -1;

    float4 sum2 = read_imagef(src, sampler, loc + (int2)(-1,-1)) *  1 +
                  read_imagef(src, sampler, loc + (int2)(-1, 0)) *  1 +
                  read_imagef(src, sampler, loc + (int2)(-1, 1)) *  1 +
                  read_imagef(src, sampler, loc + (int2)( 1,-1)) * -1 +
                  read_imagef(src, sampler, loc + (int2)( 1, 0)) * -1 +
                  read_imagef(src, sampler, loc + (int2)( 1, 1)) * -1;

    float4 dstPix = hypot(sum1, sum2) * div + bias;
    write_imagef(dst, loc, dstPix);
}

__kernel void roberts_global(__write_only image2d_t dst,
                             __read_only  image2d_t src,
                             float div,
                             float bias)
{
    const sampler_t sampler = (CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE   |
                               CLK_FILTER_NEAREST);

    int2 loc = (int2)(get_global_id(0), get_global_id(1));

    float4 sum1 = read_imagef(src, sampler, loc + (int2)(-1,-1)) *  1 +
                  read_imagef(src, sampler, loc + (int2)( 0,-1)) * -1;


    float4 sum2 = read_imagef(src, sampler, loc + (int2)(-1, 0)) * -1 +
                  read_imagef(src, sampler, loc + (int2)( 0, 0)) *  1;


    float4 dstPix = hypot(sum1, sum2) * div + bias;
    write_imagef(dst, loc, dstPix);
}


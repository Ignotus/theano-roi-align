#section support_code_apply

// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

using std::max;
using std::min;

#define Dtype float

// The following chunks are borrowed from Caffe. ------------------------------

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      return 1; \
    } \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

// ----------------------------------------------------------------------------

__global__ void APPLY_SPECIFIC(ROIAlignForward)(
    const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, Dtype* argmax_data_x,
    Dtype* argmax_data_y) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale;

    Dtype roi_width = roi_end_w - roi_start_w + 1;
    Dtype roi_height = roi_end_h - roi_start_h + 1;

    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    Dtype hstart = static_cast<Dtype>(ph) * bin_size_h;
    Dtype wstart = static_cast<Dtype>(pw) * bin_size_w;
    Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h;
    Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w;

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0.), float(height));
    hend = min(max(hend + roi_start_h, 0.), float(height));
    wstart = min(max(wstart + roi_start_w, 0.), float(width));
    wend = min(max(wend + roi_start_w, 0.), float(width));
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    Dtype maxidx_x = -1;
    Dtype maxidx_y = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (Dtype h = hstart; h < hend; h += 1.) {
      for (Dtype w = wstart; w < wend; w += 1.) {
        // Selecting four regular locations for bilinear interpolation
        Dtype x_left = floor(w);
        Dtype x_right = ceil(w);
        Dtype y_bottom = floor(h);
        Dtype y_top = ceil(h);

        int top_left_index = static_cast<int>(y_top * width + x_left);
        int top_right_index = static_cast<int>(y_top * width + x_right);
        int bottom_left_index = static_cast<int>(y_bottom * width + x_left);
        int bottom_right_index = static_cast<int>(y_bottom * width + x_right);

        bool is_top_left_in = x_left >= 0 && x_left <= width - 1
            && y_top >= 0 && y_top <= height - 1;
        bool is_top_right_in = x_right >= 0 && x_right <= width - 1
            && y_top >= 0 && y_top <= height - 1;
        bool is_bottom_left_in = x_left >= 0 && x_left <= width - 1
            && y_bottom >= 0 && y_bottom <= height - 1;
        bool is_bottom_right_in = x_right >= 0 && x_right <= width - 1
            && y_bottom >= 0 && y_bottom <= height - 1;

        Dtype val = 0;
        if (is_top_left_in)
          val += (w - x_left) * (y_top - h) * bottom_data[top_left_index];
        if (is_top_right_in)
          val += (x_right - w) * (y_top - h) * bottom_data[top_right_index];
        if (is_bottom_left_in)
          val += (w - x_left) * (h - y_bottom) * bottom_data[bottom_left_index];
        if (is_bottom_right_in)
          val += (x_right - w) * (h - y_bottom) * bottom_data[bottom_right_index];

        if (val > maxval) {
          maxval = val;
          maxidx_x = w;
          maxidx_y = h;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data_x[index] = maxidx_x;
    argmax_data_y[index] = maxidx_y;
  }
}

__global__ void APPLY_SPECIFIC(ROIAlignBackward)(
    const int nthreads, const Dtype* top_diff, const Dtype* argmax_data_x,
    const Dtype* argmax_data_y, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      // And it assumes that we don't have any negative offset of course
      int roi_start_w = floor(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = floor(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = ceil(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = ceil(offset_bottom_rois[4] * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const Dtype* offset_argmax_data_x = argmax_data_x + offset;
      const Dtype* offset_argmax_data_y = argmax_data_y + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit
      Dtype roi_width = roi_end_w - roi_start_w + 1;
      Dtype roi_height = roi_end_h - roi_start_h + 1;

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);

      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          Dtype max_x = offset_argmax_data_x[ph * pooled_width + pw];
          Dtype max_y = offset_argmax_data_y[ph * pooled_width + pw];

          int x_left = floor(max_x);
          int x_right = ceil(max_x);
          int y_bottom = floor(max_y);
          int y_top = ceil(max_y);

          if (x_left == w && y_top == h)
            gradient += (max_x - x_left) * (y_top - max_y)
                * offset_top_diff[ph * pooled_width + pw];
          else if (x_left == w && y_bottom == h)
            gradient += (max_x - x_left) * (max_y - y_bottom)
                * offset_top_diff[ph * pooled_width + pw];
          else if (x_right == w && y_top == h)
            gradient += (x_right - max_x) * (y_top - max_y)
                * offset_top_diff[ph * pooled_width + pw];
          else if (x_right == w && y_bottom == h)
            gradient += (x_right - max_x) * (max_y - y_bottom)
                * offset_top_diff[ph * pooled_width + pw];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

int APPLY_SPECIFIC(Forward_gpu)(CudaNdarray* data,
                                CudaNdarray* rois,
                                CudaNdarray** out,
                                CudaNdarray** argmaxes_x,
                                CudaNdarray** argmaxes_y) {
  int batch_size = CudaNdarray_DIMS(rois)[0];
  int channels = CudaNdarray_DIMS(data)[1];
  int height = CudaNdarray_DIMS(data)[2];
  int width = CudaNdarray_DIMS(data)[3];

  // Prepare outputs.
  int dims[] = {0, 0, 0, 0};
  dims[0] = batch_size;
  dims[1] = channels;
  dims[2] = POOLED_HEIGHT;
  dims[3] = POOLED_WIDTH;

  int count = batch_size * channels * POOLED_HEIGHT * POOLED_WIDTH;

  CudaNdarray_prep_output(out, 4, dims);
  CudaNdarray_prep_output(argmaxes_x, 4, dims);
  CudaNdarray_prep_output(argmaxes_y, 4, dims);

  APPLY_SPECIFIC(ROIAlignForward)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, data->devdata, SPATIAL_SCALE, channels, height, width,
          POOLED_HEIGHT, POOLED_WIDTH, rois->devdata, (*out)->devdata,
          (*argmaxes_x)->devdata, (*argmaxes_y)->devdata);
  CUDA_POST_KERNEL_CHECK;

  return 0;
}

int APPLY_SPECIFIC(Backward_gpu)(CudaNdarray* data,
                                 CudaNdarray* rois,
                                 CudaNdarray* argmaxes_x,
                                 CudaNdarray* argmaxes_y,
                                 CudaNdarray* out_grad,
                                 CudaNdarray** data_grad) {
  int count = CudaNdarray_SIZE(data);
  int batch_size = CudaNdarray_DIMS(rois)[0];
  int channels = CudaNdarray_DIMS(data)[1];
  int height = CudaNdarray_DIMS(data)[2];
  int width = CudaNdarray_DIMS(data)[3];

  // Prepare data grad.
  CudaNdarray_prep_output(data_grad, 4, CudaNdarray_DIMS(data));

  cudaMemset((*data_grad)->devdata, Dtype(0.), count);

  APPLY_SPECIFIC(ROIAlignBackward)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, out_grad->devdata, argmaxes_x->devdata, argmaxes_y->devdata,
          batch_size,
          SPATIAL_SCALE, channels, height, width, POOLED_HEIGHT, POOLED_WIDTH,
          (*data_grad)->devdata, rois->devdata);
  CUDA_POST_KERNEL_CHECK;

  return 0;
}

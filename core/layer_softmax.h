/*!
  The MIT License (MIT)

  Copyright (c)2016 Olivier Soares

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
 */


#ifndef CORE_LAYER_SOFTMAX_H_
#define CORE_LAYER_SOFTMAX_H_


#include <core/layer_classifier.h>
#include <memory>
#include <cmath>
#include <vector>


namespace jik {


/*!
 *  \class  LayerSoftMax
 *  \brief  Softmax
 */
template <typename Dtype>
class LayerSoftMax: public LayerClassifier<Dtype> {
  // Public types
 public:
  typedef Dtype                   Type;
  typedef LayerClassifier<Dtype>  Parent;


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name: layer name
   *  \param[in]  in  : input activations
   */
  LayerSoftMax(const char*                                     name,
               const std::vector<std::shared_ptr<Mat<Dtype>>>& in):
    Parent(name, in) {}

  /*!
   * Destructor.
   */
  virtual ~LayerSoftMax() {}

  /*!
   * Forward pass.
   * The forward pass calculates the outputs activations
   * in regard to the inputs activations and weights.
   *
   *  \param[in]  state: state
   */
  virtual void Forward(const State& state) {
    Dtype*       out_data = Parent::out_[0]->Data();
    const Dtype* in_data  = Parent::in_[0]->Data();

    uint32_t data_size = Parent::out_[0]->size[0] * Parent::out_[0]->size[1] *
                         Parent::out_[0]->size[2];
    if (!data_size) {
      return;
    }
    uint32_t batch_size = Parent::out_[0]->size[3];

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      // Find the max value for the current data
      uint32_t offset = batch * data_size;
      Dtype val_max   = in_data[offset];
      for (uint32_t i = 1; i < data_size; ++i) {
        if (in_data[offset + i] > val_max) {
          val_max = in_data[offset + i];
        }
      }

      // out = norm(exp(in - max))
      Dtype sum = static_cast<Dtype>(0);
      for (uint32_t i = 0; i < data_size; ++i) {
        out_data[offset + i] = std::exp(in_data[offset + i] - val_max);
        sum                 += out_data[offset + i];
      }
      sum = static_cast<Dtype>(1) / sum;
      for (uint32_t i = 0; i < data_size; ++i) {
        out_data[offset + i] *= sum;
      }
    }
  }

  /*!
   * Backward pass.
   * The backward pass calculates the inputs activations and weights
   * derivatives in regard to the outputs activations derivatives.
   *
   *  \param[in]  state: state
   */
  virtual void Backward(const State& state) {
    const Dtype* out_data      = Parent::out_[0]->Data();
    Dtype*       in_deriv_data = Parent::in_[0]->DerivData();
    const Dtype* label_data    = Parent::in_[1]->Data();

    uint32_t data_size  = Parent::out_[0]->size[0] * Parent::out_[0]->size[1] *
                          Parent::out_[0]->size[2];
    uint32_t batch_size = Parent::out_[0]->size[3];

    Parent::in_[0]->deriv->data = Parent::out_[0]->data;
    Parent::loss_               = static_cast<Dtype>(0);
    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      uint32_t index = batch * data_size +
                       static_cast<uint32_t>(label_data[batch]);
      in_deriv_data[index] -= static_cast<Dtype>(1);
      Parent::loss_        -= std::log(out_data[index]);
    }
  }
};


}  // namespace jik


#endif  // CORE_LAYER_SOFTMAX_H_

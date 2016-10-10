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


#ifndef CORE_SOLVER_RMSPROP_H_
#define CORE_SOLVER_RMSPROP_H_


#include <core/solver.h>
#include <memory>
#include <limits>


namespace jik {


/*!
 *  \class  SolverRMSprop
 *  \brief  RMSprop solver
 */
template <typename Dtype>
class SolverRMSprop: public Solver<Dtype> {
  // Public types
 public:
  typedef Dtype         Type;
  typedef Solver<Dtype> Parent;


  // Protected attributes
 protected:
  Dtype decay_rate_;      // Decay rate
  Dtype reg_strength_;    // L2 regularization strength
  Dtype gradient_clip_;   // Gradient clipping value


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  print_each   : print the model stats every n steps
   *  \param[in]  test_each    : test the model every n steps
   *  \param[in]  save_each    : save the model every n steps
   *  \param[in]  lr_scale_each: save the model every n steps
   *  \param[in]  lr_scale     : learning rate scale
   *  \param[in]  decay_rate   : decay rate
   *  \param[in]  reg_strength : L2 regularization strength
   *  \param[in]  gradient_clip: gradient clipping value
   */
  SolverRMSprop(uint32_t print_each, uint32_t test_each, uint32_t save_each,
                uint32_t lr_scale_each, Dtype lr_scale,
                Dtype decay_rate, Dtype reg_strength, Dtype gradient_clip):
    Parent(print_each, test_each, save_each, lr_scale_each, lr_scale) {
    decay_rate_    = decay_rate;
    reg_strength_  = reg_strength;
    gradient_clip_ = gradient_clip;
  }

  /*!
   * Destructor.
   */
  virtual ~SolverRMSprop() {}

  /*!
   * Stochastic gradient descent.
   *
   *  \param[in]  weight       : weights
   *  \param[in]  weight_prev  : previous weights
   *  \param[in]  learning_rate: learning rate
   *  \param[in]  batch_size   : batch size
   *  \param[in]  decay_rate   : decay rate
   *  \param[in]  reg_strength : L2 regularization strength
   *  \param[in]  gradient_clip: gradient clipping value
   */
  static void RMSprop(const std::shared_ptr<Mat<Dtype>>& weight,
                      const std::shared_ptr<Mat<Dtype>>& weight_prev,
                      Dtype learning_rate, uint32_t batch_size,
                      Dtype decay_rate, Dtype reg_strength,
                      Dtype gradient_clip) {
  Dtype* weight_data             = weight->Data();
  const Dtype* weight_deriv_data = weight->DerivData();
  Dtype* weight_prev_data        = weight_prev->Data();

  for (uint32_t i = 0; i < weight->Size(); ++i) {
      // RMSprop adaptive learning rate
      Dtype dv            = weight_deriv_data[i] / batch_size;
      weight_prev_data[i] = decay_rate * weight_prev_data[i] +
                            (static_cast<Dtype>(1) - decay_rate) * dv * dv;

      // Gradient clip
      if (dv > gradient_clip) {
        dv = gradient_clip;
      } else if (dv < -gradient_clip) {
        dv = -gradient_clip;
      }

      // Update and regularize
      weight_data[i] += -learning_rate * (dv / std::sqrt(weight_prev_data[i] +
        std::numeric_limits<Dtype>::epsilon())) -
        reg_strength * weight_data[i];
    }
  }

  /*!
   * Learning function.
   *
   *  \param[in]  model        : model
   *  \param[in]  learning_rate: learning rate
   */
  virtual void Learn(Model<Dtype>* model, Dtype learning_rate) const {
    if (!model) {
      return;
    }
    for (size_t i = 0; i < Parent::weight_.size(); ++i) {
      RMSprop(Parent::weight_[i], Parent::weight_prev_[i], learning_rate,
              model->BatchSize(), decay_rate_, reg_strength_, gradient_clip_);
    }
    model->ClearDeriv();
  }
};


}  // namespace jik


#endif  // CORE_SOLVER_RMSPROP_H_

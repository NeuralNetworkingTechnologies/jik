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


#ifndef CORE_LAYER_CLASSIFIER_H_
#define CORE_LAYER_CLASSIFIER_H_


#include <core/layer.h>
#include <core/log.h>
#include <memory>
#include <vector>


namespace jik {


/*!
 *  \class  LayerClassifier
 *  \brief  Classifier base class
 */
template <typename Dtype>
class LayerClassifier: public Layer<Dtype> {
  // Public types
 public:
  typedef Dtype         Type;
  typedef Layer<Dtype>  Parent;


  // Protected attributes
 protected:
  Dtype loss_;  // Loss value


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name: layer name
   *  \param[in]  in  : input activations
   */
  LayerClassifier(const char*                                     name,
                  const std::vector<std::shared_ptr<Mat<Dtype>>>& in):
    Parent(name, in) {
    // Make sure we have 2 inputs (input itself and labels)
    Check(Parent::in_.size() == 2, "Layer '%s' must have 2 inputs",
          Parent::Name());

    // Initialize the loss to zero
    loss_ = static_cast<Dtype>(0);

    // Create 1 output, same size as the inputs
    Parent::out_.resize(1);
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(Parent::in_[0]->size);
  }

  /*!
   * Destructor.
   */
  virtual ~LayerClassifier() {}

  /*!
   * Get the loss value.
   *
   *  \return Loss value
   */
  Dtype Loss() const {
    return loss_;
  }
};


}  // namespace jik


#endif  // CORE_LAYER_CLASSIFIER_H_

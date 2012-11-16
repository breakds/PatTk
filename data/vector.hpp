/*********************************************************************************
 * File: vector.hpp
 * Author: BreakDS <breakds@cs.wisc.edu>
 *         University of Wisconsin-Madison, 2012
 * Description: Provide the operations on vectors, which is represented as
 *              c++ arrays
 *********************************************************************************/

#pragma once

#include <cmath>
#include <type_traits>
#include "LLPack/utils/candy.hpp"

namespace PatTk
{

  template <typename numeric>
  class epsilon
  {
  public:
    static const numeric value;
  };

  template <>
  const int epsilon<int>::value(1);

  template <>
  const unsigned char epsilon<unsigned char>::value(1);

  template <>
  const float epsilon<float>::value(1e-5);

  template <>
  const double epsilon<double>::value(1e-5);

  
  
  

  /* note: this function works even if src and dst are the same */
  template<typename floating>
  inline void normalize_vec( const floating *src, floating *dst, int dim )
  {
    // std::static_assert( std::is_floating_point<floating>::valye >,
    //                     "template parameter floating is not a floating point type." );
    
    floating tmp = 0;
    for ( int i=0; i<dim; i++ ) {
      tmp += src[i] * src[i];
    }
    tmp += epsilon<floating>::value * epsilon<floating>::value;
    tmp = sqrt( tmp );
    for ( int i=0; i<dim; i++ ) {
      dst[i] = src[i] / tmp;
    }
  }

  template <typename dataType>
  inline double norm_l2( const dataType* vec, int dim )
  {
    double re = 0;
    for ( int i=0; i<dim; i++ ) {
      re += vec[i] * vec[i];
    }
    return sqrt(re);
  }

}

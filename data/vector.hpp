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
    static const numeric v1;
  };

  template <>
  const int epsilon<int>::value(1);

  template <>
  const int epsilon<int>::v1(1);

  template <>
  const unsigned char epsilon<unsigned char>::value(1);

  template <>
  const unsigned char epsilon<unsigned char>::v1(1);

  template <>
  const float epsilon<float>::value(1e-5);

  template <>
  const float epsilon<float>::v1(1e-1);

  template <>
  const double epsilon<double>::value(1e-5);

  template <>
  const double epsilon<double>::v1(1e-1);

  
  

  /* note: this function works even if src and dst are the same */
  template<typename floating>
  inline void normalize_vec( const floating *src, floating *dst, int dim, floating norm=1.0 )
  {
    
    floating tmp = 0;
    for ( int i=0; i<dim; i++ ) {
      tmp += src[i] * src[i];
    }

    tmp += epsilon<floating>::v1 * epsilon<floating>::v1;

    tmp = sqrt( tmp ) / norm;
    
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


  template <typename dataType>
  inline double dist_l2( const dataType* vec0, const dataType *vec1, int dim )
  {
    const dataType *vp0 = vec0;
    const dataType *vp1 = vec1;
    double re = 0;
    for ( int i=0; i<dim; i++ ) {
      typename Generalized<dataType>::type tmp = *(vp0++) - *(vp1++);
      re += tmp * tmp;
    }
    return sqrt(re);
  }

  

  /* scale a vector */
  template <typename dataType>
  inline void scale( dataType* vec, int dim, dataType wt )
  {
    dataType* vecp = vec;
    for ( int i=0; i<dim; i++ ) *(vecp++) *= wt;
  }

  /* add two vectors and assign the result to the first one */
  template <typename dataType>
  inline void addto( dataType *v0, const dataType *v1, int dim )
  {
    dataType* vp0 = v0;
    const dataType* vp1 = v1;
    for ( int i=0; i<dim; i++ ) *(vp0++) += *(vp1++);
  }

  /* add two vectors and store the result in the third parameter */
  template <typename dataType>
  inline void add( const dataType *v0, const dataType *v1, dataType *res, int dim )
  {
    const dataType* vp0 = v0;
    const dataType* vp1 = v1;
    dataType *resp = res;
    for ( int i=0; i<dim; i++ ) *(resp++) = *(vp0++) + *(vp1++);
  }

  /* combine two vectors with weights alpha and beta, store the reuslt
     in the third parameter */
  template <typename dataType>
  inline void combine( const dataType *v0, const dataType *v1, dataType *res, int dim,
                       const dataType alpha, const dataType beta )
  {
    const dataType* vp0 = v0;
    const dataType* vp1 = v1;
    dataType *resp = res;
    for ( int i=0; i<dim; i++ ) *(resp++) = *(vp0++) * alpha + *(vp1++) * beta;
  }

  /* shift the vector as a histogram where the shift bins is delta > 0 */
  template <typename dataType>
  inline void shift( dataType *vec, int dim, float delta )
  {
    dataType tmp[dim];
    memcpy( tmp, vec, sizeof( dataType ) * dim );
    int lo = static_cast<int>( delta );
    float alpha = delta - lo;
    float balpha = 1.0 - alpha;
    for ( int i=0; i<dim; i++ ) {
      if ( lo >= dim ) lo -= dim;
      int hi = lo + 1;
      if ( hi >= dim ) hi -= dim;
      vec[i] = tmp[lo] * balpha + tmp[hi] * alpha;
      lo++;
    }
  }
  
}

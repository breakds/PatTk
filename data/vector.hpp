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
#include <string>
#include "LLPack/utils/extio.hpp"
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


  template<typename dataType>
  inline void zero( dataType *target, int dim )
  {
    memset( target, 0, sizeof(dataType) * dim );
  }


  template<typename dataType>
  inline void copy( dataType *dst, dataType *src, int dim )
  {
    memcpy( dst, src, sizeof(dataType) * dim );
  }
  

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
  inline double norm2( const dataType* vec, int dim )
  {
    double re = 0;
    for ( int i=0; i<dim; i++ ) {
      re += vec[i] * vec[i];
    }
    return re;
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


  /* scale a vector and store the result in the last parameter */
  template <typename dataType>
  inline void scale( const dataType* vec, int dim, dataType wt, dataType *res )
  {
    const dataType* vecp = vec;
    dataType *resp = res;
    for ( int i=0; i<dim; i++ ) *(resp++) = *(vecp++) * wt;
  }

  /* add two vectors and assign the result to the first one */
  template <typename dataType>
  inline void addto( dataType *v0, const dataType *v1, int dim )
  {
    dataType* vp0 = v0;
    const dataType* vp1 = v1;
    for ( int i=0; i<dim; i++ ) *(vp0++) += *(vp1++);
  }

  /* add scaled v1 to v0 */
  template <typename dataType>
  inline void addScaledTo( dataType *v0, const dataType *v1, int dim, dataType wt )
  {
    dataType* vp0 = v0;
    const dataType* vp1 = v1;
    for ( int i=0; i<dim; i++ ) *(vp0++) += wt * (*(vp1++));
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

  /* get the difference between two vectors and store the result in
     the third parameter */
  template <typename dataType>
  inline void minus( const dataType *v0, const dataType *v1, dataType *res, int dim )
  {
    const dataType* vp0 = v0;
    const dataType* vp1 = v1;
    dataType *resp = res;
    for ( int i=0; i<dim; i++ ) *(resp++) = *(vp0++) - *(vp1++);
  }

  /* minus v1 from v0 */
  template <typename dataType>
  inline void minusFrom( dataType *v0, const dataType *v1, int dim )
  {
    dataType* vp0 = v0;
    const dataType* vp1 = v1;
    for ( int i=0; i<dim; i++ ) *(vp0++) -= *(vp1++);
  }

  /* minus scaled v1 from v0 */
  template <typename dataType>
  inline void minusScaledFrom( dataType *v0, const dataType *v1, int dim, dataType wt )
  {
    dataType* vp0 = v0;
    const dataType* vp1 = v1;
    for ( int i=0; i<dim; i++ ) *(vp0++) -= wt * (*(vp1++));
  }

  /* negate v0 */
  template <typename dataType>
  inline void negate( dataType *v0, int dim ) {
    for ( int i=0; i<dim; i++ ) v0[i] = -v0[i];
  }

  /* combine two vectors with weights alpha and beta, store the reuslt
     in the third parameter */
  template <typename dataType>
  inline void combine( const dataType *v0, const dataType *v1, dataType *res, int dim,
                       float alpha, float beta )
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


  /* print vector */
  void printVec( const double *x, int dim ) {
    printf( "( " );
    for ( int i=0; i<dim; i++ ) {
      printf( "%.4lf ", x[i] );
    }
    printf( ")\n" );
  }
  void printVec( const float *x, int dim ) {
    printf( "( " );
    for ( int i=0; i<dim; i++ ) {
      printf( "%.4f ", x[i] );
    }
    printf( ")\n" );
  }
  void printVec( const int *x, int dim ) {
    printf( "( " );
    for ( int i=0; i<dim; i++ ) {
      printf( "%d ", x[i] );
    }
    printf( ")\n" );
  }
  void printVec( const unsigned char *x, int dim ) {
    printf( "( " );
    for ( int i=0; i<dim; i++ ) {
      printf( "%hhu ", x[i] );
    }
    printf( ")\n" );
  }
  
  
  /* sum the vector */
  template <typename dataType>
  inline dataType sum_vec( const dataType *x, int dim )
  {
    dataType sum = 0;
    for ( int i=0; i<dim; i++ ) {
      sum += x[i];
    }
    return sum;
  }

  /* Calculate the entropy based on the distribution */
  template <typename dataType>
  inline double entropy( const dataType *vec, int dim )
  {
    double s = static_cast<double>( sum_vec( vec, dim ) );
    double re = 0.0;
    const dataType *x = vec;
    for ( int i=0; i<dim; i++ ) {
      double v = *(x++) / s;
      if ( v >= 1e-6 ) {
        re += v * log( v );
      }
    }
    return -re;
  }



  /*
   * watershed algorithm for simplex constraints projection
   * min_x |x-y|^2
   * s.t. x >= 0
   *      sum x = 1
   */
  template <typename dataType>
  inline void watershed( const dataType *y, dataType *x, int dim ) 
  {
    // Projection: sum to 1
    dataType sum = 0.0;
    for ( int i=0; i<dim; i++ ) sum += y[i];
    dataType p = (1.0 - sum) / dim;
    dataType _y[dim];
    for ( int i=0; i<dim; i++ ) _y[i] = y[i] + p;


    // Projection: All >= 0
    p = 0.0f;
    int index[dim];
    int k = 0;
    for ( int i=0; i<dim; i++ ) {
      if ( _y[i] < 0.0 ) {
        p -= _y[i];
        x[i] = 0.0;
      } else {
        index[k++] = i;
      }
    }
    
    // sort
    int tmp;
    for ( int i=0; i<k-1; i++ ) {
      for ( int j=i+1; j<k; j++ ) {
        if ( _y[index[j]] < _y[index[i]]  ) {
          tmp = index[j];
          index[j] = index[i];
          index[i] = tmp;
        }
      }
    }


    dataType tc = 0.0;
    dataType bt = 0.0;
    int i = 0;
    for ( i=0; i<k; i++ ) {
      _y[index[i]] -= bt;
      tc = _y[index[i]] * (k-i);
      if ( tc < p ) {
        p -= tc;
        bt += _y[index[i]];
        x[index[i]] = 0.0;
      } else {
        break;
      }
    }

  
    tc = p / (k-i);
    x[index[i]] = _y[index[i]] - tc;
    i++;
    for ( ; i<k; i++ ) {
      x[index[i]] = _y[index[i]] - bt - tc;
    }
  }

}

/*********************************************************************************
 * File: kernel.hpp
 * Description: Kernels (detail implementations) for Random Trees.
 * by BreakDS, University of Wisconsin Madison, Sat Jul 14 07:58:16 CDT 2012
 *********************************************************************************/

#pragma once

#include <memory>
#include <type_traits>
#include <cstdlib>
#include "LLPack/utils/extio.hpp"
#include "LLPack/algorithms/sort.hpp"
#include "data/2d.hpp"
#include "data/features.hpp"
#include "tree.hpp"


// TODO: temp options
#define projDim 10
#define hypoNum 20
#define convergeTh 120

namespace PatTk
{
  template <typename cellType, typename valueType>
  class BasicKernel : public AbstractKernel<cellType,valueType>
  {
  public:
    // the branching class (test function)
    class branch : public AbstractBranch<cellType,valueType>
    {
    public:
      vector<int> proj;
      vector<typename BasicKernel<cellType,valueType>::data_t> vertex[2];
      int th;
      // TODO: add score so that don't need to calculate again

    private:

      // prohibited copy constructor and copy assignment operator
      branch( const branch& other );
      const branch& operator=( const branch& other );
      
    public:
      template<typename T=typename cellType::type>
      inline int calculate( const typename BasicKernel<cellType,valueType>::patch_t& patch,
                            typename std::enable_if<std::is_integral<T>::value>::type
                            __attribute__((__unused__)) *padding=0 ) const
      {
        int val = 0;
        for ( int i=0, end=static_cast<int>( proj.size() ); i<end; i++ ) {
          int tmp = vertex[0][i] - patch[proj[i]];
          val += tmp * tmp;
          tmp = vertex[1][i] - patch[proj[i]];
          val -= tmp * tmp;
        }
        return val;
      }

      // below are two calculate functions for integral types

      template<typename T=typename cellType::type>
      inline int calculate( const typename BasicKernel<cellType,valueType>::patch_t& a,
                            const typename BasicKernel<cellType,valueType>::patch_t& b,
                            typename std::enable_if<std::is_integral<T>::value>::type
                            __attribute__((__unused__)) *padding=0 ) const
      {
        int val = 0;
        for ( int i=0, end=static_cast<int>( proj.size() ); i<end; i++ ) {
          int tmp = a[proj[i]] - b[proj[i]];
          val += tmp * tmp;
        }
        return val;
      }

      template<typename T=typename cellType::type>
      inline double calculate( const typename BasicKernel<cellType,valueType>::patch_t& patch,
                               typename std::enable_if<std::is_floating_point<T>::value>::type
                               __attribute__((__unused__)) *padding=0 ) const
      {
        double val = 0;
        for ( int i=0, end=static_cast<int>( proj.size() ); i<end; i++ ) {
          double tmp = vertex[0][i] - patch[proj[i]];
          val += tmp * tmp;
          tmp = vertex[1][i] - patch[proj[i]];
          val -= tmp * tmp;
        }
        return val;
      }

      // below are tow caculate functions for floating point types

      template<typename T=typename cellType::type>
      inline double calculate( const typename BasicKernel<cellType,valueType>::patch_t& a,
                               const typename BasicKernel<cellType,valueType>::patch_t& b,
                               typename std::enable_if<std::is_floating_point<T>::value>::type
                               __attribute__((__unused__)) *padding=0 ) const
      {
        double val = 0;
        for ( int i=0, end=static_cast<int>( proj.size() ); i<end; i++ ) {
          double tmp = a[proj[i]] - b[proj[i]];
          val += tmp * tmp;
        }
        return val;
      }

      inline bool isValid( typename std::enable_if<std::is_integral<typename cellType::type>::value>::type
                           __attribute__((__unused__)) *padding=0 ) const
      {

        int val = 0;
        for ( int i=0, end=static_cast<int>( proj.size() ); i<end; i++ ) {
          int tmp = static_cast<int>( vertex[0][i] ) - static_cast<int>( vertex[1][i] );
          val += tmp * tmp;
        }
        if ( val < convergeTh ) {
          return false;
        }
        return true;
      }


    public:
      
      branch() {}
      
      branch( branch&& other ) noexcept
      {
        proj.swap( other.proj );
        vertex[0].swap( other.vertex[0] );
        vertex[1].swap( other.vertex[1] );
        th = other.th;
      }

      const branch& operator=( branch&& other ) noexcept
      {
        proj.swap( other.proj );
        vertex[0].swap( other.vertex[0] );
        vertex[1].swap( other.vertex[1] );
        th = other.th;
        return (*this);
      }

      int operator()( const typename BasicKernel<cellType,valueType>::patch_t& patch ) const
      {
        if ( calculate(patch) < th  ) {
          return 0;
        }
        return 1;
      }
    };

  public:
    static vector<branch> RaiseHypothesis( const vector<typename BasicKernel<cellType,valueType>::patch_t>& patchList,
                                           const int* ref,
                                           int len )
    {
      int dim = patchList[0].dim();
      
      vector<branch> hypos;
      hypos.clear();
      for ( int i=0; i<hypoNum; i++ ) {
        hypos.push_back( std::move( branch()) );
        branch& b = hypos[i];
        // TODO: use randperm()
        b.proj.resize( projDim );
        for ( int j=0; j<projDim; j++ ) {
          b.proj[j] = rand() % dim;
        }

        // TODO: use randperm()
        int pos[2] = {0,0};
        pos[0] = rand() % len;
        do { pos[1] = rand() % len; } while ( pos[1] == pos[0] );
        for ( int k=0; k<2; k++ ) {
          b.vertex[k].resize( projDim );
          for ( int j=0; j<projDim; j++ ) {
            b.vertex[k][j] = patchList[ref[pos[k]]][b.proj[j]];
          }
        }

        // get median value
        vector<int> vals;
        vals.clear();
        for ( int j=0; j<len; j++ ) {
          vals.push_back( b.calculate( patchList[ref[j]] ) );
        }

        b.th = sorting::median( vals );

      }
      return hypos;
    }

    static double ScoreHypothesis( const vector<typename BasicKernel<cellType,valueType>::patch_t>& patchList,
                                   const int* ref,
                                   int len,
                                   const branch& hypo )
    {
      vector<int> nearest;
      nearest.resize( len );
      for ( int i=0; i<len; i++ ) {
        int min = -1;
        nearest[i] = -1;
        for ( int j=0; j<len; j++ ) {
          if ( i == j )  continue;
          int dist = hypo.calculate( patchList[ref[i]], patchList[ref[j]] );
          if ( -1 == nearest[i] || dist < min ) {
            min = dist;
            nearest[i] = j;
          }
        }
      }

      vector<int> tag;
      tag.resize( len );
      for ( int i=0; i<len; i++ ) {
        tag[i] = hypo( patchList[ref[i]] );
      }

      int accuErr = 0;
      for ( int i=0; i<len; i++ ) {
        if ( tag[i] != tag[nearest[i]] ) {
          accuErr++;
        }
      }
      
      return static_cast<double>( -accuErr );
    }

    static bool terminate( const vector<typename BasicKernel<cellType,valueType>::patch_t>
                           __attribute__((__unused__)) &patchList,
                           const int __attribute__((__unused__)) *ref,
                           int len )
    {
      if ( len < 5 ) return true;
      return false;
    }
    
  };
}

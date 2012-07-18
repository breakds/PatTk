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
#include "LLPack/utils/candy.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/algorithms/sort.hpp"
#include "data/2d.hpp"
#include "data/features.hpp"
#include "tree.hpp"

using namespace EnvironmentVariable;

// TODO: temp options

namespace PatTk
{
  template <typename cellType, typename valueType>
  class BasicKernel : public AbstractKernel<cellType,valueType>
  {
  public:
    struct Options
    {
      int projDim;
      int hypoNum;
      int convergeTh;
      int set_size_a;
      int set_size_b;
    };

  public:

    /// the branching class (test function)
    class branch : public AbstractBranch<cellType,valueType>
    {
    public:
      vector<int> proj;
      vector<typename BasicKernel<cellType,valueType>::data_t> vertex[2];
      typename BasicKernel<cellType,valueType>::gen_data_t th;
      // TODO: add score so that don't need to calculate again
    private:
      // prohibit calling copy constructor and copy assignment operator
      branch( const branch& other );
      const branch& operator=( const branch& other );


    public:
      inline typename BasicKernel<cellType,valueType>::gen_data_t
      calculate( const typename BasicKernel<cellType,valueType>::patch_t& patch ) const
      {
        typename BasicKernel<cellType,valueType>::gen_data_t val(0);
        for ( int i=0, end=static_cast<int>( proj.size() ); i<end; i++ ) {
          typename BasicKernel<cellType,valueType>::gen_data_t tmp = vertex[0][i] - patch[proj[i]];
          val += tmp * tmp;
          tmp = vertex[1][i] - patch[proj[i]];
          val -= tmp * tmp;
        }
        return val;
      }

      inline typename BasicKernel<cellType,valueType>::gen_data_t
      calculate( const typename BasicKernel<cellType,valueType>::patch_t& a,
                 const typename BasicKernel<cellType,valueType>::patch_t& b) const
      {
        typename BasicKernel<cellType,valueType>::gen_data_t val = 0;
        for ( int i=0, end=static_cast<int>( proj.size() ); i<end; i++ ) {
          typename BasicKernel<cellType,valueType>::gen_data_t tmp = a[proj[i]] - b[proj[i]];
          val += tmp * tmp;
        }
        return val;
      }
      
      inline bool isValid() const
      {

        typename BasicKernel<cellType,valueType>::gen_data_t val = 0;
        for ( int i=0, end=static_cast<int>( proj.size() ); i<end; i++ ) {
          typename BasicKernel<cellType,valueType>::gen_data_t tmp = vertex[0][i] - vertex[1][i];
          val += tmp * tmp;
        }

        if ( val < env["converge-th"] ) {
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

      // debugging:
      void trace()
      {
        for ( uint i=0; i<proj.size(); i++ ) {
          printf( "%3hhu\t", vertex[0][i] );
        }
        printf( "\n" );
        for ( uint i=0; i<proj.size(); i++ ) {
          printf( "%3hhu\t", vertex[1][i] );
        }
        printf( "\n" );

      }

    };


  public:
    
    static int split( const vector<typename BasicKernel<cellType,valueType>::patch_t>& patches,
                      const int* ref,
                      int len,
                      branch& fork )
    {
      // 1. sub-sampling set A and set B. For each patch in set A,
      // search for its nearest neighbor in set B.

      int dim = patches[0].dim();
        
      vector<int> A = std::move( rndgen::randperm<int>( ref, env["set-size-a"], len ) );
      vector<int> B = std::move( rndgen::randperm<int>( ref, env["set-size-b"], len ) );

      int sizeA = static_cast<int>( A.size() );
      int sizeB = static_cast<int>( B.size() );
       
      
      vector<typename BasicKernel<cellType,valueType>::gen_data_t> vals;
      vals.resize( len );

      int minErr = -1;

      // 2. enumerating hypothesis
      for ( int i=0; i<env["hypo-num"]; i++ ) {
        branch b;
        b.proj = std::move( rndgen::randperm( dim, env["proj-dim"] ) );
        int pos[2] = {0,0};
        pos[0] = rand() % len;
        do { pos[1] = rand() % len; } while ( 1 < len && pos[1] == pos[0] );
        for ( int k=0; k<2; k++ ) {
          b.vertex[k].resize( env["proj-dim"] );
          for ( int j=0; j<env["proj-dim"]; j++ ) {
            b.vertex[k][j] = patches[ref[pos[k]]][b.proj[j]];
          }
        }
        
        typename BasicKernel<cellType,valueType>::gen_data_t min = 0;
        typename BasicKernel<cellType,valueType>::gen_data_t max = 0;
        for ( int j=0; j<len; j++ ) {
          typename BasicKernel<cellType,valueType>::gen_data_t val = b.calculate( patches[ref[j]] );
          vals[j] = val;
          if ( 0 == j || val < min ) min = val;
          if ( 0 == j || val > max ) max = val;
        }


        
        b.th = sorting::median( vals );
        
        // compensating for the case where median happens to be the
        // max or min which contaminates the split
        if ( b.th == max ) {
          b.th--;
        } else if ( b.th == min ) {
          b.th++;
        }

        if ( !b.isValid() ) continue;

        // scoring
        int error = 0;
        for ( int j=0; j<sizeA; j++ ) {
          int nearest = B[0];
          typename BasicKernel<cellType,valueType>::gen_data_t min = b.calculate( patches[A[j]], patches[B[0]] );
          for ( int k=1; k<sizeB; k++ ) {
            typename BasicKernel<cellType,valueType>::gen_data_t dist = b.calculate( patches[A[j]], patches[B[k]] );
            if ( dist < min ) {
              min = dist;
              nearest = B[k];
            }
          }
          if ( b(patches[A[j]]) != b(patches[nearest]) ) {
            error++; 
          }
        }

        if ( -1 == minErr || error < minErr ) {
          minErr = error;
          fork = std::move( b );
        }
        
      }
      
      if ( -1 == minErr ) {
        return -1;
      }
      
      return 0;
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

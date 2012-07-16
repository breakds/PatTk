/*********************************************************************************
 * File: tree.hpp
 * Description: Random Tree Utilities. Designed for working with patches.
 * by BreakDS, University of Wisconsin Madison, Sat Jul 14 07:58:16 CDT 2012
 *********************************************************************************/

#pragma once

#include <utility>
#include <memory>
#include <deque>
#include <tuple>
#include <type_traits>
#include "data/2d.hpp"
#include "data/features.hpp"

namespace PatTk
{
  template <typename cellType, typename valueType>
  class AbstractBranch
  {
  public:
    virtual int operator()( const typename Image<cellType,valueType>::Patch& patch ) const = 0;
  };

  
  
  template <typename cellType, typename valueType>
  class AbstractKernel
  {
    static_assert( std::is_base_of< AbstractCell<typename cellType::type>, cellType >::value,
                   "cellType is not a valid cell type. (does not derive from AbstractCell." );
  public:
    // Useful type interface
    typedef cellType cell_t;
    typedef valueType value_t;
    typedef typename Image<cellType,valueType>::Patch patch_t;
    typedef typename cellType::type data_t;
  public:
    static vector<AbstractBranch<cellType,valueType> > RaiseHypothesis() {}
  };


  
  
  template <typename kernel>
  class Tree
  {
    static_assert( std::is_base_of< AbstractKernel<typename kernel::cell_t, typename kernel::value_t>,
                                    kernel>::value, "kernel does derive from AbstractKernel." );
                   
  private:
    // Children
    std::unique_ptr<Tree<kernel> > child[2];
    // Test Function
    typename kernel::branch fork;
    // patches (valid for leaf nodes only)
    vector<typename kernel::patch_t> patches;
    
  private:
    // Prohibited copy/move constructors
    Tree( const Tree<kernel>& tree ) {}
    Tree( Tree<kernel>&& tree ) {}
    // Prohibited copy/move assignment
    const Tree<kernel>& operator=( const Tree<kernel>& tree ) {}
    const Tree<kernel>& operator=( Tree<kernel>&& tree ) {}

  public:

    // default constructor
    Tree() : fork()
    {
      child[0].reset( nullptr );
      child[1].reset( nullptr );
      patches.clear();
    }

    Tree( const vector<typename kernel::patch_t>& patchList )
    {
      grow( patchList );
    }

    // Constructor 1: Build Tree from scractch
    void grow( const vector<typename kernel::patch_t>& patchList )
    {
      // The references to patches
      int indices[patchList.size()];
      for ( uint i=0; i<patchList.size(); i++ ) indices[i] = i;

      // The stack of triplets:
      // ( node*, reference, length )
      std::deque<std::tuple<Tree<kernel>*,int*,int> > stack;
      stack.clear();
      //      stack.push_back( std::move( std::make_tuple( this, indices, static_cast<int>(patchList.size()) ) ) );
      stack.push_back( std::tuple<Tree<kernel>*,int*,int>( this, indices, static_cast<int>(patchList.size()) ) );
      
      while( !stack.empty() ) {

        // Get next node from stack
        Tree<kernel> *node = std::get<0>( stack.front() );
        int *ref = std::get<1>( stack.front() );
        int len = std::get<2>( stack.front() );
        stack.pop_front();

        // whether satisfies the terminating condition
        if ( kernel::terminate( patchList, ref, len ) ) {
          for ( int i=0; i<len; i++ ) {
            node->patches.push_back( patchList[ref[i]] );
          }
          continue;
        }

        
        vector<typename kernel::branch> hypos = std::move( kernel::RaiseHypothesis( patchList, ref, len ) );
        assert( hypos.size() > 0 );

        // Acquire the best hypothesis
        double best = 0.0;
        int bestHypoIdx = -1;
        
        for ( int i=0, end=static_cast<int>( hypos.size() ); i<end; i++ ) {
          if ( hypos[i].isValid() ) {
            double score = kernel::ScoreHypothesis( patchList, ref, len, hypos[i] );
            if ( -1 == bestHypoIdx || score > best ) {
              best = score;
              bestHypoIdx = i;
            }
          }
        }

        // whether satisfies the covergence condition
        if ( -1 == bestHypoIdx ) {
          for ( int i=0; i<len; i++ ) {
            node->patches.push_back( patchList[ref[i]] );
          }
          continue;
        }

        
        // Acquire the best hypothesis
        node->fork = std::move( hypos[bestHypoIdx] );
        
        // Get the right branch's reference start point
        int right(-1), t(0);
        for ( int i=0; i<len; i++ ) {
          if ( 0 == node->fork( patchList[ref[i]] ) ) {
            right++;
            t = ref[i];
            ref[i] = ref[right];
            ref[right] = t;
          }
        }
        right++;

        assert( right != 0 && right != len );
        node->child[0].reset( new Tree() );
        node->child[1].reset( new Tree() );
        stack.push_back( std::make_tuple( node->child[0].get(), ref, right ) );
        stack.push_back( std::make_tuple( node->child[1].get(), ref + right, len - right ) );
      }
    }
    
    inline bool leaf() const
    {
      if ( nullptr == child[0] && nullptr == child[1] ) {
        return true;
      }
      return false;
    }

    inline const Tree<kernel>* direct( const typename kernel::patch_t& query ) const
    {
      if ( leaf() ) {
        return this;
      }
      
      Tree<kernel> *node = child[fork(query)].get();
      
      while ( !node->leaf() ) {
        node = node->child[node->fork( query )].get();
      }
      
      return node;
    }


    inline void Summary() const
    {
      if ( leaf() ) {
        printf( "leaf with \n" );
        for ( auto& ele : patches ) {
          ele.Summary();
        }
      } else {
        printf( "Internal Node.\n" );
      }
    }
  };
    
};

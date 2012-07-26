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
#include "LLPack/algorithms/sort.hpp"

namespace PatTk
{
  template <typename cellType, typename valueType>
  class AbstractBranch
  {
  public:
    virtual int operator()( const typename Image<cellType,valueType>::Patch& patch ) const = 0;
    virtual void write( FILE* out ) const = 0;
    virtual void read( FILE* in ) = 0;
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
    typedef typename Generalized<typename cellType::type>::type gen_data_t;
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
    
  private:
    // Prohibited copy/move constructors
    Tree( const Tree<kernel>& tree ) {}
    Tree( Tree<kernel>&& tree ) {}
    // Prohibited copy/move assignment
    const Tree<kernel>& operator=( const Tree<kernel>& tree ) {}
    const Tree<kernel>& operator=( Tree<kernel>&& tree ) {}

  public:
    // patches (valid for leaf nodes only)
    vector<typename kernel::patch_t> patches;


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

    // write()
    void write( const std::string& filename ) const
    {
      WITH_OPEN( out, filename.c_str(), "w" );
      write( out );
      END_WITH( out );
    }
    
    void write( FILE *out ) const
    {
      if ( leaf() ) {
        int flag = 1;
        fwrite( &flag, sizeof(int), 1, out );
        // write leaf patches
        int len = static_cast<int>( patches.size() );
        fwrite( &len, sizeof(int), 1, out );
        for ( auto& ele : patches ) {
          ele.write( out );
        }
      } else {
        int flag = 0;
        fwrite( &flag, sizeof(int), 1, out );
        // write branch function
        fork.write( out );
        // write child
        child[0]->write( out );
        child[1]->write( out );
      }
    }

    // read()
    template <typename cellType, typename valueType, bool lite>
    void read( const std::string& filename, const Album<cellType,valueType,lite>& album )
    {
      WITH_OPEN( in, filename.c_str(), "r" );
      read( in, album );
      END_WITH( in );
    }

    // read()
    template <typename cellType, typename valueType, bool lite>
    void read( FILE* in, const Album<cellType,valueType,lite>& album )
    {
      int flag = 0;
      fread( &flag, sizeof(int), 1, in );

      if ( 1 == flag ) {
        // read leaf patches
        int len = 0;
        fread( &len, sizeof(int), 1, in );
        patches.clear();
        for ( int i=0; i<len; i++ ) {
          patches.push_back( album.ReadPatch( in ) );
        }
      } else {
        // read branch function
        fork.read( in );
        
        child[0].reset( new Tree() );
        child[1].reset( new Tree() );
        child[0]->read( in, album );
        child[1]->read( in, album );
      }
      
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


        if ( kernel::terminate( patchList, ref, len ) ) {
          // first termination condition statisfied
          for ( int i=0; i<len; i++ ) {
            node->patches.push_back( patchList[ref[i]] );
          }
          continue;
        }


        if ( -1 == kernel::split( patchList, ref, len, node->fork ) ) {
          // second termination condition satisfied
          for ( int i=0; i<len; i++ ) {
            node->patches.push_back( patchList[ref[i]] );
          }
          continue;
        }

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
        printf( "leaf with %ld patches. They are:\n", patches.size() );
        for ( auto& ele : patches ) {
          ele.Summary();
        }
      } else {
        printf( "Internal Node.\n" );
      }
    }
  };
    
};

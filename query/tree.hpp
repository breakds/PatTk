/*********************************************************************************
 * File: tree.hpp
 * Description: Random Tree Utilities. Designed for working with patches.
 * by BreakDS, University of Wisconsin Madison, Fri Nov 16 16:04:26 CST 2012
 *********************************************************************************/

#pragma once

#include <vector>
#include <deque>
#include <memory>
#include "Revolver.hpp"

namespace PatTk
{

  struct LocInfo
  {
    int id, y, x;

    inline void write( FILE *out )
    {
      fwrite( &id, sizeof(int), 1, out );
      fwrite( &y, sizeof(int), 1, out );
      fwrite( &x, sizeof(int), 1, out );
    }

    inline void read( FILE *in )
    {
      fread( &id, sizeof(int), 1, in );
      fread( &y, sizeof(int), 1, in );
      fread( &x, sizeof(int), 1, in );
    }
    
  };


  template <typename T>
  class SimpleKernel
  {
  public:
    typedef T dataType;
    
    class State
    {
    public:
      int *idx;
      int len;
      Shuffler shuffler;

      State( int *i, int l, int s )
        : idx(i), len(l), shuffler(s) {}
      
      State( int *i, int l, const Shuffler& s )
        : idx(i), len(l), shuffler(s) {}
    };

    class Judger
    {
    public:
      dataType th;
      int component;

      inline void write( FILE *out )
      {
        fwrite( &th, sizeof(dataType), 1, out );
        fwrite( &component, sizeof(int), 1, out );
      }

      inline void read( FILE *in )
      {
        fread( &th, sizeof(dataType), 1, in );
        fread( &component, sizeof(int), 1, in );
      }
      
      inline int operator()( typename FeatImage<T>::PatchProxy &p )
      {
        if ( p(component) < th ) return 0;
        return 1;
      }

      inline int operator()( T* p )
      {
        if ( p[component] < th ) return 0;
        return 1;
      }
    };

    static int numHypo;
    static int stopNum;
    static typename Generalized<dataType>::type converge;
    
    static int split( std::vector<typename FeatImage<T>::PatchProxy> &list, State& state,
                      Judger &judger )
    {

      if ( state.len <= stopNum ) {
        return -1;
      }
      if ( 0 == state.shuffler.Number() ) {
        return -2;
      }

      state.shuffler.ResetShuffle();

      int trial = 0;
      uint c[numHypo];
      while ( SHUFFLER_ERROR != ( c[trial] = state.shuffler.Next() ) && trial < numHypo ) {
        typename Generalized<T>::type min = list[state.idx[0]](c[trial]);
        typename Generalized<T>::type max = list[state.idx[0]](c[trial]);
        for ( int i=1; i<state.len; i++ ) {
          if ( list[state.idx[i]](c[trial]) > max ) {
            max = list[state.idx[i]](c[trial]);
          } else if ( list [state.idx[i]](c[trial]) < min ) {
            min = list[state.idx[i]](c[trial]);
          }
        }

        if ( max - min < converge ) {
          state.shuffler.Disqualify();
        } else {
          trial++;
        }
      }

      if ( 0 == trial ) {
        return -3;
      }

      int minDiff = -1;
      for ( int t=0; t<trial; t++ ) {
        int pick = rand() % state.len;
        dataType th = list[state.idx[pick]](c[t]);
        int leftNum = 0;
        int rightNum = 0;
        for ( int i=0; i<state.len; i++ ) {
          if ( list[state.idx[i]](c[t]) < th ) {
            leftNum++;
          } else {
            rightNum++;
          }
        }
        if ( -1 == minDiff || abs( leftNum - rightNum ) < minDiff ) {
          minDiff = abs( leftNum - rightNum );
          judger.th = th;
          judger.component = c[t];
        }
      }

      if ( -1 == minDiff ) {
        return -4;
      }

      if ( state.len == minDiff ) {
        return -5;
      }
      
      int right = -1;
      for ( int i=0; i<state.len; i++ ) {
        if ( 0 == judger( list[state.idx[i]] ) ) {
          right++;
          int tmp = state.idx[right];
          state.idx[right] = state.idx[i];
          state.idx[i] = tmp;
        }
      }

      return right;
      
    }
  };

  template <typename T>
  int SimpleKernel<T>::numHypo = 10;

  template <typename T>
  int SimpleKernel<T>::stopNum = 10;

  template <>
  double SimpleKernel<float>::converge = 0.05;

  template <>
  double SimpleKernel<double>::converge = 0.05;

  template <>
  int SimpleKernel<unsigned char>::converge = 10;



  


  
  template <typename kernel>
  class Tree
  {
  private:
    std::unique_ptr< Tree<kernel> > child[2];
    std::vector<LocInfo> store;
    typename kernel::Judger judger;
    
  public:

    Tree() 
    {
      child[0].reset( nullptr );
      child[1].reset( nullptr );
      store.clear();
    }


    static std::unique_ptr<Tree<kernel> > read( std::string filename )
    {
      std::unique_ptr<Tree<kernel> > tree;
      WITH_OPEN( in, filename.c_str(), "r" );
      tree.reset( new Tree( in ) );
      END_WITH( in );
      return tree;
    }
    
    
    Tree( FILE *in )
    {
      judger.read( in );
      unsigned char finished = 0;
      fread( &finished, sizeof(unsigned char), 1, in );
      if ( 1 == finished ) {
        int len = 0;
        fread( &len, sizeof(int), 1, in );
        store.resize( len );
        for ( int i=0; i<len; i++ ) {
          store[i].read( in );
        }
      } else {
        child[0].reset( new Tree( in ) );
        child[1].reset( new Tree( in ) );
      }
    }



    Tree( std::vector<typename FeatImage<typename kernel::dataType>::PatchProxy> &list, int* idx, int len )
    {
      store.clear();
      
      std::deque<std::pair<typename kernel::State,Tree<kernel>*> > stack;
      stack.push_back( std::make_pair( typename kernel::State( idx, len, list[0].dim() ), this ) );
      
      while ( !stack.empty() ) {
        typename kernel::State &state = stack.front().first;
        Tree<kernel> *node = stack.front().second;
        int right = kernel::split( list, state, node->judger );
        if ( right >= 0 ) {
          // internal node
          node->child[0].reset( new Tree() );
          node->child[1].reset( new Tree() );
          stack.push_back( std::make_pair( typename kernel::State( state.idx,
                                                                   right,
                                                                   state.shuffler ),
                                           node->child[0].get() ));
          stack.push_back( std::make_pair( typename kernel::State( state.idx + right,
                                                                   state.len - right,
                                                                   state.shuffler ),
                                           node->child[1].get() ));
        } else {
          // leaf node
          for ( int i=0; i<state.len; i++ ) {
            LocInfo loc;
            loc.y = list[state.idx[i]].y;
            loc.x = list[state.idx[i]].x;
            loc.id = list[state.idx[i]].parent->id;
            node->store.push_back( loc );
          }
        }
        stack.pop_front();
      }
    }

    void write( std::string filename )
    {
      WITH_OPEN( out, filename.c_str(), "w" );
      write( out );
      END_WITH( out );
    }
    
    void write( FILE *out )
    {
      judger.write( out );
      if ( isLeaf() ) {
        unsigned char uc = 1;
        fwrite( &uc, sizeof(unsigned char), 1, out );
        int len = static_cast<int>( store.size() );
        fwrite( &len, sizeof(int), 1, out );
        for ( auto& ele : store ) {
          ele.write( out );
        }
      } else {
        unsigned char uc = 0;
        fwrite( &uc, sizeof(unsigned char), 1, out );
        child[0]->write( out );
        child[1]->write( out );
      }
    }

    inline bool isLeaf()
    {
      if ( child[0] ) return false;
      return true;
    }

    std::vector<LocInfo>& query( typename FeatImage<typename kernel::dataType>::PatchProxy &p )
    {
      if ( isLeaf() ) {
        return store;
      } else {
        return child[judger(p)]->query( p );
      }
    }
  };
}

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
#include "../data/Label.hpp"
#include "../data/FeatImage.hpp"

namespace PatTk
{

  template <typename kernel> class Tree; // forward declaration

  struct LocInfo
  {
    int id, y, x;
    
    LocInfo( int id1, int y1, int x1 ) : id(id1), y(y1), x(x1) {}
    
    inline void write( FILE *out ) const
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

    inline LocInfo( FILE *in )
    {
      fread( &id, sizeof(int), 1, in );
      fread( &y, sizeof(int), 1, in );
      fread( &x, sizeof(int), 1, in );
    }
  };

  template <typename kernel>
  struct NodeInfo
  {
  public:
    Tree<kernel> * node;
    int leafID;
    
    NodeInfo( Tree<kernel> *n = nullptr, int l = -1 ) : node(n), leafID(l) {}

    inline NodeInfo( FILE *in )
    {
      fread( &leafID, sizeof(int), 1, in );
    }

    inline void write( FILE *out ) const
    {
      fwrite( &leafID, sizeof(int), 1, out );
    }
  };


  struct LeafInfo
  {

  public:
    std::vector<LocInfo> store;
    std::vector<float> q; // label probablity map for center pixel

    inline LeafInfo()
    {
      store.clear();
      q.clear();
    }

    inline LeafInfo( FILE *in )
    {
      store.clear();
      int len = 0;
      fread( &len, sizeof(int), 1, in );
      for ( int i=0; i<len ;i++ ) {
        store.push_back( LocInfo( in ) );
      }
      
      fread( &len, sizeof(int), 1, in );
      assert( 0 == len || LabelSet::classes == len );
      float tmp;
      q.resize( len );
      for ( int i=0; i<len; i++ ) {
        fread( &tmp, sizeof(float), 1, in );
        q[i] = tmp;
      }
    }

    inline LeafInfo( LeafInfo &&other )
    {
      store.swap( other.store );
      q.swap( other.q );
    }

    inline LeafInfo( const LeafInfo &other )
    {
      store = other.store;
      q = other.q;
    }


    inline int size() const
    {
      return static_cast<int>( store.size() );
    }

    inline const LeafInfo& operator=( LeafInfo &&other )
    {
      store.swap( other.store );
      q.swap( other.q );
      return (*this);
    }
    

    inline void add( int id, int y, int x )
    {
      store.push_back( LocInfo( id, y, x ) );
    }

    inline void write( FILE *out ) const
    {
      int len = static_cast<int>( store.size() );
      fwrite( &len, sizeof(int), 1, out );
      for ( auto& ele : store ) {
        ele.write( out );
      }
      
      len = static_cast<int>( q.size() );
      fwrite( &len, sizeof(int), 1, out );
      for ( int i=0; i<len; i++ ) {
        fwrite( &q[i], sizeof(float), 1, out );
      }
    }

    inline void read( FILE *in )
    {
      store.clear();
      int len = 0;
      fread( &len, sizeof(int), 1, in );
      for ( int i=0; i<len ;i++ ) {
        store.push_back( LocInfo( in ) );
      }
      fread( &len, sizeof(int), 1, in );
      assert( 0 == len || LabelSet::classes == len );
      float tmp;
      q.resize( len );
      for ( int i=0; i<len; i++ ) {
        fread( &tmp, sizeof(float), 1, in );
        q[i] = tmp;
      }
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
      int depth;

      State( int *i, int l, int s, int d )
        : idx(i), len(l), shuffler(s), depth(d) {}
      
      State( int *i, int l, const Shuffler& s, int d )
        : idx(i), len(l), shuffler(s), depth(d) {}
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
      
      inline int operator()( const typename FeatImage<T>::PatchProxy &p ) const
      {
        if ( p(component) < th ) return 0;
        return 1;
      }

      inline int operator()( const T* p ) const
      {
        if ( p[component] < th ) return 0;
        return 1;
      }
    };

    static int numHypo;
    static int stopNum;
    static typename Generalized<dataType>::type converge;

    static int split( const std::vector<typename FeatImage<T>::PatchProxy> &list, State& state,
                      Judger &judger, int max_depth = -1 )
    {
      
      if ( state.len <= stopNum ) {
        return -1;
      }
      if ( 0 == state.shuffler.Number() ) {
        return -2;
      }

      if ( max_depth == state.depth ) {
        return -6;
      }

      state.shuffler.ResetShuffle();
      
      int trial = 0;
      uint c[numHypo];
      dataType th[numHypo];
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
          dataType range = max - min;
          th[trial] = rand() / static_cast<dataType>( RAND_MAX ) * range * 0.95 + range * 0.025 + min;
          trial++;
        }
      }
      
      if ( 0 == trial ) {
        return -3;
      }

      int minDiff = -1;
      for ( int t=0; t<trial; t++ ) {
        int leftNum = 0;
        int rightNum = 0;
        for ( int i=0; i<state.len; i++ ) {
          if ( list[state.idx[i]](c[t]) < th[t] ) {
            leftNum++;
          } else {
            rightNum++;
          }
        }
        if ( -1 == minDiff || abs( leftNum - rightNum ) < minDiff ) {
          minDiff = abs( leftNum - rightNum );
          judger.th = th[t];
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

      return right + 1;
      
    }
  };

  template <typename T>
  int SimpleKernel<T>::numHypo = 10;

  template <typename T>
  int SimpleKernel<T>::stopNum = 1;

  template <>
  double SimpleKernel<float>::converge = 0.001;

  template <>
  double SimpleKernel<double>::converge = 0.001;

  template <>
  int SimpleKernel<unsigned char>::converge = 10;

  

  


  
  template <typename kernel>
  class Tree
  {
  private:
    std::unique_ptr< Tree<kernel> > child[2];
    typename kernel::Judger judger;
    
  public:
    int nodeID;
    
    
    Tree() 
    {
      child[0].reset( nullptr );
      child[1].reset( nullptr );
      nodeID = -1;
    }


    static std::unique_ptr<Tree<kernel> > read( std::string filename, std::vector<NodeInfo<kernel> >& nodes )
    {
      std::unique_ptr<Tree<kernel> > tree;
      WITH_OPEN( in, filename.c_str(), "r" );
      tree.reset( new Tree( in, nodes ) );
      END_WITH( in );
      return tree;
    }
    
    
    Tree( FILE *in, std::vector<NodeInfo<kernel> >& nodes )
    {
      judger.read( in );
      unsigned char finished = 0;
      fread( &finished, sizeof(unsigned char), 1, in );
      fread( &nodeID, sizeof(int), 1, in );
      nodes[nodeID].node = this;
      if ( 1 != finished ) {
        child[0].reset( new Tree( in, nodes ) );
        child[1].reset( new Tree( in, nodes ) );
      }
    }

    

    
    Tree( const std::vector<typename FeatImage<typename kernel::dataType>::PatchProxy> &list, int* idx, int len,
          std::vector<NodeInfo<kernel> >& nodes, std::vector<LeafInfo> &leaves, int max_depth=-1 )
    {

      std::deque<std::pair<typename kernel::State,Tree<kernel>*> > stack;
      stack.push_back( std::make_pair( typename kernel::State( idx, len, list[0].dim(), 0 ), this ) );
      
      while ( !stack.empty() ) {
        typename kernel::State &state = stack.front().first;
        Tree<kernel> *node = stack.front().second;
        int right = kernel::split( list, state, node->judger, max_depth );
        if ( right >= 0 ) {
          // internal node
          node->child[0].reset( new Tree() );
          node->child[1].reset( new Tree() );
          
          
          stack.push_back( std::make_pair( typename kernel::State( state.idx,
                                                                   right,
                                                                   state.shuffler,
                                                                   state.depth + 1 ),
                                           node->child[0].get() ));
          stack.push_back( std::make_pair( typename kernel::State( state.idx + right,
                                                                   state.len - right,
                                                                   state.shuffler,
                                                                   state.depth + 1 ),
                                           node->child[1].get() ));
          // register NodeInfo
#         pragma omp critical
          {
            node->nodeID = static_cast<int>( nodes.size() );
            nodes.emplace( nodes.end(), node );
          }
        } else {
          // leaf node
          int leafID = 0;
#         pragma omp critical
          {
            leafID = static_cast<int>( leaves.size() );
            leaves.emplace( leaves.end() );
            node->nodeID = static_cast<int>( nodes.size() );
            nodes.emplace( nodes.end(), node, leafID );
            for ( int i=0; i<state.len; i++ ) {
              leaves[leafID].add( list[state.idx[i]].id(),
                                  list[state.idx[i]].y,
                                  list[state.idx[i]].x );
            }
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
        fwrite( &nodeID, sizeof(int), 1, out );
      } else {
        unsigned char uc = 0;
        fwrite( &uc, sizeof(unsigned char), 1, out );
        fwrite( &nodeID, sizeof(int), 1, out );
        child[0]->write( out );
        child[1]->write( out );
      }
    }

    inline bool isLeaf() const
    {
      return ( nullptr == child[0] ) && ( nullptr == child[1] );
    }

    inline const std::unique_ptr<Tree<kernel> >& getChild( int index )
    {
      return child[index];
    }

    int query( const typename FeatImage<typename kernel::dataType>::PatchProxy &p ) const
    {
      if ( isLeaf() ) {
        return nodeID;
      } else {
        return child[judger(p)]->query( p );
      }
    }

    int query_node( const typename FeatImage<typename kernel::dataType>::PatchProxy &p, int depth ) const
    {
      if ( isLeaf() || 0 == depth ) {
        return nodeID;
      } else {
        return child[judger(p)]->query_node( p, depth - 1 );
      }
    }

    int query( const typename kernel::dataType *p ) const
    {
      if ( isLeaf() ) {
        return nodeID;
      } else {
        return child[judger(p)]->query( p );
      }
    }

    int query_node( const typename kernel::dataType *p, int depth ) const
    {
      if ( isLeaf() || 0 == depth ) {
        return nodeID;
      } else {
        return child[judger(p)]->query_node( p, depth - 1 );
      }
    }

  private:
    void collect( std::vector<int> &store ) const
    {
      if ( isLeaf() ) {
        store.push_back( nodeID );
      } else {
        child[0]->collect( store );
        child[1]->collect( store );
      }
    }
  public:
    std::vector<int> collect() const
    {
      std::vector<int> store;
      collect( store );
      return store;
    }

    int maxDepth() const
    {
      if ( isLeaf() ) {
        return 1;
      } else {
        int i = child[0]->maxDepth();
        int j = child[1]->maxDepth();
        if ( i > j ) {
          return i + 1;
        } else {
          return j + 1;
        }
      }
    }
  };
}

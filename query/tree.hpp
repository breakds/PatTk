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
#include "kernels/SimpleKernel.hpp"
#include "kernels/EntropyKernel.hpp"
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
        // right is the starting index of the right branch
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


    Tree( const std::vector<typename FeatImage<typename kernel::dataType>::PatchProxy> &list,
          const std::vector<typename FeatImage<float>::PatchProxy> &labels,
          int* idx, int len,
          std::vector<NodeInfo<kernel> >& nodes, std::vector<LeafInfo> &leaves, int max_depth=-1 )
    {

      std::deque<std::pair<typename kernel::State,Tree<kernel>*> > stack;
      stack.push_back( std::make_pair( typename kernel::State( idx, len, list[0].dim(), 0 ), this ) );
      
      while ( !stack.empty() ) {
        typename kernel::State &state = stack.front().first;
        Tree<kernel> *node = stack.front().second;
        int right = kernel::split( list, labels, state, node->judger, max_depth );

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

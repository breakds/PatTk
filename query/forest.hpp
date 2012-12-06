/*********************************************************************************
 * File: forest.hpp
 * Description: Random Forest wrapper for a set of random trees
 * by BreakDS, University of Wisconsin Madison, Wed Dec  5 13:54:06 CST 2012
 *********************************************************************************/

#pragma once
#include <sys/stat.h>
#include "LLPack/algorithms/random.hpp"
#include "tree.hpp"


namespace PatTk
{

  template <typename kernel>
  class Forest
  {
  private:
    std::vector<std::unique_ptr<Tree<kernel> > > trees;
    std::vector<LeafInfo> leaves;

  public:
    
    Forest( int n,
            const std::vector<typename FeatImage<typename kernel::dataType>::PatchProxy> &list,
            float proportion = 1.1f )
    {
      trees.resize( n );
      std::vector<std::vector<LeafInfo> > leaf( n );

      int len = static_cast<int>( list.size() );
      int trueLen = len;
      if ( proportion < 1.0f ) trueLen = static_cast<int>( len * proportion );
      
      int **idx = new int*[n];
      
      for ( int i=0; i<n; i++ ) {
        leaf[i].clear();
        rndgen::randperm( len, trueLen, idx[i] );
        trees[i].reset( new Tree<kernel>( list, idx[i], trueLen, leaf[i] ) );
      }

      // Merging leaf arrays
      int leafCount = 0;
      int offset[n];
      for ( int i=0; i<n; i++ ) {
        offset[i] = leafCount;
        leafCount += static_cast<int>( leaf[i].size() );
      }

      leaves.resize( leafCount );
      
      for ( int i=0; i<n; i++ ) {
        trees[i]->OffsetLeafID( offset[i] );
        for ( int j=0, p=offset[i], end=static_cast<int>( leaf[i].size() ); j<end; j++, p++ ) {
          leaves[p] = std::move( leaf[i][j] );
        }
      }
      
      for ( int i=0; i<n; i++ ) delete[] idx[i];
      delete[] idx;
    }

    /* ---------- I/O ---------- */

    Forest( std::string dir )
    {

      trees.clear();
      int i = 0;
      do {
        struct stat fileAtt;
        
        if ( 0 != stat( strf( "%s/tree.%d", dir.c_str(), i ).c_str() , &fileAtt ) ) {
          trees.push_back( new Tree<kernel>( strf( "%s/tree.%d", dir.c_str(), i ).c_str() ) );
        } else {
          break;
        }
      } while (true);

      leaves.clear();
      
      WITH_OPEN( in, strf( "%s/leaves.dat", dir.c_str() ).c_str(), "r" );
      int len = 0;
      fread( &len, sizeof(int), 1, in );
      for ( int i=0; i<len; i++ ) {
        leaves.push_back( std::move( LeafInfo( in ) ) );
      }
      END_WITH( in );

      Done( "Forest loaded (%ld trees, %ld leaves).", trees.size(), leaves.size() );
    }

    void write( std::string dir ) const
    {
      system( strf( "mkdir -p %s", dir.c_str() ).c_str() );
      for ( int i=0; i<static_cast<int>( trees.size() ); i++ ) {
        trees[i]->write( strf( "%s/tree.%d", dir.c_str(), i ).c_str() );
      }

      WITH_OPEN( out, strf( "%s/leaves.dat", dir.c_str() ).c_str(), "w" );
      int len = static_cast<int>( leaves.size() );
      fwrite( &len, sizeof(int), 1, out );
      for ( int i=0; i<len; i++ ) {
        leaves[i].write( out );
      }
      END_WITH( out );
    }

    /* ---------- Accessor ---------- */
    inline const LeafInfo& operator()( const int index )
    {
      return leaves[index];
    }
    

    /* ---------- Query ---------- */
    inline std::vector<int> query( const typename FeatImage<typename kernel::dataType>::PatchProxy &p ) const
    {
      std::vector<int> res;
      res.reserve( trees.size() );

      for ( auto& ele : trees ) {
        res.push_back( ele->query( p ) );
      }
    }

    inline std::vector<int> query( const typename kernel::dataType *p ) const
    {
      std::vector<int> res;
      res.reserve( trees.size() );

      for ( auto& ele : trees ) {
        res.push_back( ele->query( p ) );
      }
    }

    


    int size() const
    {
      return static_cast<int>( trees.size() );
    }

  };
  
}

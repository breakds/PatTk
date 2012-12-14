/*********************************************************************************
 * File: forest.hpp
 * Description: Random Forest wrapper for a set of random trees
 * by BreakDS, University of Wisconsin Madison, Wed Dec  5 13:54:06 CST 2012
 *********************************************************************************/

#pragma once
#include <iostream>
#include <unordered_map>
#include "LLPack/algorithms/random.hpp"
#include "../data/Label.hpp"
#include "tree.hpp"


namespace PatTk
{

  template <typename kernel>
  class Forest
  {
  private:
    std::vector<std::unique_ptr<Tree<kernel> > > trees;
    std::vector<LeafInfo> leaves;
    std::vector<std::unordered_map<int,int> > weights;
    
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

      weights.clear();
      
      for ( int i=0; i<n; i++ ) delete[] idx[i];
      delete[] idx;
    }

    /* ---------- I/O ---------- */

    Forest( std::string dir )
    {
      
      trees.clear();
      int i = 0;
      do {

        std::ifstream fin( strf( "%s/tree.%d", dir.c_str(), i ) );
        if ( fin.good() ) {
          trees.push_back( Tree<kernel>::read( strf( "%s/tree.%d", dir.c_str(), i ).c_str() ) );
        } else {
          break;
        }
        i++;
      } while (true);

      leaves.clear();
      
      WITH_OPEN( in, strf( "%s/leaves.dat", dir.c_str() ).c_str(), "r" );
      int len = 0;
      fread( &len, sizeof(int), 1, in );
      for ( int i=0; i<len; i++ ) {
        leaves.push_back( std::move( LeafInfo( in ) ) );
      }
      END_WITH( in );

      weights.clear();
      readWeights( dir );

      Done( "Forest loaded (%ld trees, %ld leaves).", trees.size(), leaves.size() );
    }


    inline void writeWeights( std::string dir ) const
    {
      if ( 0 < weights.size() ) {
        WITH_OPEN( out, strf( "%s/weights.dat", dir.c_str() ).c_str(), "w" );
        int len = static_cast<int>( leaves.size() );
        fwrite( &len, sizeof(int), 1, out );
        for ( int i=0; i<len; i++ ) {
          len = static_cast<int>( weights[i].size() );
          fwrite( &len, sizeof(int), 1, out );
          for ( auto& ele : weights[i] ) {
            fwrite( &ele.first, sizeof(int), 1, out );
            fwrite( &ele.second, sizeof(int), 1, out );
          }
        }
        END_WITH( out );
      }
    }

    inline void readWeights( std::string dir )
    {
      std::ifstream fin( strf( "%s/weights.dat", dir.c_str() ) );
      if ( fin.good() ) {
        WITH_OPEN( in, strf( "%s/weights.dat", dir.c_str() ).c_str(), "r" );
        int len = 0;
        fread( &len, sizeof(int), 1, in );
        weights.resize( len );
        for ( int i=0; i<len; i++ ) {
          weights[i].clear();
          fread( &len, sizeof(int), 1, in );
          for ( int j=0; j<len; j++ ) {
            int first = 0;
            int second = 0;
            fread( &first, sizeof(int), 1, in );
            fread( &second, sizeof(int), 1, in );
            weights[i][first] = second;
          }
        }
        END_WITH( in );
      }
    }

    inline void writeLeaves( std::string dir ) const
    {
      WITH_OPEN( out, strf( "%s/leaves.dat", dir.c_str() ).c_str(), "w" );
      int len = static_cast<int>( leaves.size() );
      fwrite( &len, sizeof(int), 1, out );
      for ( int i=0; i<len; i++ ) {
        leaves[i].write( out );
      }
      END_WITH( out );
    }

    inline void write( std::string dir ) const
    {
      system( strf( "mkdir -p %s", dir.c_str() ).c_str() );
      for ( int i=0; i<static_cast<int>( trees.size() ); i++ ) {
        trees[i]->write( strf( "%s/tree.%d", dir.c_str(), i ).c_str() );
      }

      writeLeaves( dir );

      writeWeights( dir );

    }

      

    /* ---------- Templates ---------- */
    inline const LeafInfo& operator()( const int index )
    {
      return leaves[index];
    }

    
    template <typename floating>
    inline void updateLabelMap( const int index, const floating *vec  )
    {
      leaves[index].q.resize( LabelSet::classes );
      for ( int i=0; i<LabelSet::classes; i++ ) {
        leaves[index].q[i] = vec[i];
      }
    }
    
    

    /* ---------- Query ---------- */

    template <typename T>
    inline std::vector<int> query( const T p ) const
    {
      static_assert( std::is_same<T,typename kernel::dataType*>::value ||
                     std::is_same<T,typename FeatImage<typename kernel::dataType>::PatchProxy&>::value,
                     "T is not a feature descriptor type." );
      
      std::vector<int> res;
      res.reserve( trees.size() );

      for ( auto& ele : trees ) {
        res.push_back( ele->query( p ) );
      }
      
      return res;
    }

    template <typename T>
    inline std::vector< std::pair<int,double> > query_with_coef( const T p ) const
    {
      static_assert( std::is_same<T,typename kernel::dataType*>::value ||
                     std::is_same<T,typename FeatImage<typename kernel::dataType>::PatchProxy&>::value,
                     "T is not a feature descriptor type." );
      
      std::vector<std::pair<int,double> > res;
      res.reserve( trees.size() );

      double w = 1.0 / trees.size();
      for ( auto& ele : trees ) {
        res.push_back( std::make_pair( ele->query( p ), w ) );
      }

      return res;
    }

    
    template <typename T>
    inline std::vector<LocInfo> pull( const T p ) const
    {
      static_assert( std::is_same<T,typename kernel::dataType*>::value ||
                     std::is_same<T,typename FeatImage<typename kernel::dataType>::PatchProxy&>::value,
                     "T is not a feature descriptor type." );
      
      std::vector<LocInfo> res;

      for ( auto& ele : trees ) {
        int lid = ele->query( p );
        for ( auto& loc : leaves[lid].store ) {
          res.push_back( loc );
        }
      }
      return res;
    }


    /* ---------- Weights ---------- */
    inline void ResetWeights()
    {
      weights.resize( leaves.size() );
    }
    
    inline void PrepareWeitghts()
    {
      if ( 0 == weights.size() ) {
        ResetWeights();
      }
    }

    template <typename T>
    inline void learn( const T p )
    {
      static_assert( std::is_same<T,typename kernel::dataType*>::value ||
                     std::is_same<T,typename FeatImage<typename kernel::dataType>::PatchProxy&>::value,
                     "T is not a feature descriptor type." );
      std::vector<int> ids;
      for ( auto& ele : trees ) {
        ids.push_back( ele->query( p ) );
      }
      
      int n = static_cast<int>( trees.size() );

      for ( int i=0; i<n; i++ ) {
        for ( int j=0; j<n; j++ ) {
          auto iter = weights[ids[i]].find(ids[j]);
          if ( weights[ids[i]].end() == iter ) {
            weights[ids[i]][ids[j]] = 1;
          } else {
            weights[ids[i]][ids[j]]++;
          }
        }
      }
    }

    inline int GetWeight( int i, int j )
    {
      assert( 0 < weights.size() );
      if ( weights[i].end() == weights[i].find( j ) ) {
        return 0;
      } else {
        int res = weights[i][j];
        return res;
      }
    }

    inline const std::unordered_map<int,int>& GetWeights( int i ) const
    {
      assert( 0 < weights.size() );
      return weights[i];
    }



    /* ---------- Properties ---------- */

    inline int size() const
    {
      return static_cast<int>( trees.size() );
    }

    inline int centers() const
    {
      return static_cast<int>( leaves.size() );
    }

  };
  
}

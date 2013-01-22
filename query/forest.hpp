/*********************************************************************************
 * File: forest.hpp
 * Description: Random Forest wrapper for a set of random trees
 * by BreakDS, University of Wisconsin Madison, Wed Dec  5 13:54:06 CST 2012
 *********************************************************************************/

#pragma once
#include <iostream>
#include <unordered_map>
#include <omp.h>
#include "LLPack/algorithms/random.hpp"
#include "LLPack/algorithms/heap.hpp"
#include "LLPack/utils/extio.hpp"
#include "../data/Label.hpp"
#include "../graph/Trans.hpp"
#include "tree.hpp"


namespace PatTk
{

  template <typename kernel>
  class Forest
  {
  private:
    std::vector<std::unique_ptr<Tree<kernel> > > trees;
    std::vector<NodeInfo<kernel> > nodes;
    std::vector<LeafInfo> leaves;
    std::vector<std::unordered_map<int,int> > weights;

  public:

    Forest( int n,
            const std::vector<typename FeatImage<typename kernel::dataType>::PatchProxy> &list,
            float proportion = 1.1f )
    {
      trees.resize( n );

      leaves.clear();
      nodes.clear();

      int len = static_cast<int>( list.size() );
      int trueLen = len;
      if ( proportion < 1.0f ) trueLen = static_cast<int>( len * proportion );
      
      int **idx = new int*[n];


      int finished = 0;
#     pragma omp parallel for num_threads(7)
      for ( int i=0; i<n; i++ ) {
        rndgen::randperm( len, trueLen, idx[i] );
        trees[i].reset( new Tree<kernel>( list, idx[i], trueLen, nodes, leaves ) );
#       pragma omp critical
        {
          progress( ++finished, n, "Tree Growth." );
        }
      }
      printf( "\n" );
      
      weights.clear();

      for ( int i=0; i<n; i++ ) delete[] idx[i];
      delete[] idx;
    }

    Forest( int n, Album<typename kernel::dataType>& album, float proportion = 1.1f ) :
      Forest( n, album.list(), proportion ) {}
      
    

    
    /* ---------- I/O ---------- */

    Forest( std::string dir )
    {

      WITH_OPEN( in, strf( "%s/nodes.dat", dir.c_str() ).c_str(), "r" );
      int len = 0;
      fread( &len, sizeof(int), 1, in );
      for ( int i=0; i<len; i++ ) {
        nodes.emplace( nodes.end(), in );
      }
      
      // dfs
      END_WITH( in );
      
      trees.clear();
      int i = 0;
      do {

        std::ifstream fin( strf( "%s/tree.%d", dir.c_str(), i ) );
        if ( fin.good() ) {
          trees.push_back( Tree<kernel>::read( strf( "%s/tree.%d", dir.c_str(), i ).c_str(), nodes ) );
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
        leaves.emplace( leaves.end(), in );
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

    inline void writeNodes( std::string dir ) const
    {
      WITH_OPEN( out, strf( "%s/nodes.dat", dir.c_str() ).c_str(), "w" );
      int len = static_cast<int>( nodes.size() );
      fwrite( &len, sizeof(int), 1, out );
      for ( int i=0; i<len; i++ ) {
        nodes[i].write( out );
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

      writeNodes( dir );
      
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

    // query leaf
    template <typename T>
    inline std::vector<int> query( const T p ) const
    {
      static_assert( std::is_same<T,typename kernel::dataType*>::value ||
                     std::is_same<T,typename FeatImage<typename kernel::dataType>::PatchProxy&>::value,
                     "T is not a feature descriptor type." );
      
      std::vector<int> res;
      res.reserve( trees.size() );

      for ( auto& ele : trees ) {
        res.push_back( nodes[ele->query( p )].leafID );
      }
      
      return res;
    }

    // query leaf
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
        res.push_back( std::make_pair( nodes[ele->query( p )].leafID, w ) );
      }

      return res;
    }


    // query node
    template <typename T>
    inline std::vector<int> query_node( const T p, int max_depth ) const
    {
      static_assert( std::is_same<T,typename kernel::dataType*>::value ||
                     std::is_same<T,typename FeatImage<typename kernel::dataType>::PatchProxy&>::value,
                     "T is not a feature descriptor type." );
      
      std::vector<int> res;
      res.reserve( trees.size() );

      for ( auto& ele : trees ) {
        res.push_back( ele->query_node( p, max_depth ) );
      }
      
      return res;
    }


    // query leaf
    template <typename T>
    inline std::vector< std::pair<int,double> > query_node_with_coef( const T p, int max_depth ) const
    {
      static_assert( std::is_same<T,typename kernel::dataType*>::value ||
                     std::is_same<T,typename FeatImage<typename kernel::dataType>::PatchProxy&>::value,
                     "T is not a feature descriptor type." );
      
      std::vector<std::pair<int,double> > res;
      res.reserve( trees.size() );

      double w = 1.0 / trees.size();
      for ( auto& ele : trees ) {
        res.push_back( std::make_pair( ele->query( p, max_depth ), w ) );
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
        int lid = nodes[ele->query( p )].leafID;
        for ( auto& loc : leaves[lid].store ) {
          res.push_back( loc );
        }
      }
      return res;
    }


    inline std::pair<int,int> getChildren( int nodeID )
    {
      if ( nodes[nodeID].node->isLeaf() ) {
        return std::make_pair( -1, -1 );
      } else {
        return std::make_pair( nodes[nodeID].node->getChild(0)->nodeID,
                               nodes[nodeID].node->getChild(1)->nodeID );
      }
    }
    
    
    GeoMap PatchMatch( const FeatImage<typename kernel::dataType> &img, 
                       int numProp = 10,
                       int maxIter = 10 )
    {
      GeoMap geomap( img.rows, img.cols );
      int dim = img.GetPatchDim();
      float feat[dim];
      float feat_c[dim];

      
      int count = 0;
      int area = img.rows * img.cols;


      heap<double, Geometric> ranker( numProp );

      for ( int i=0; i<img.rows; i++ ) {
        for ( int j=0; j<img.cols; j++ ) {

          ranker.resize( numProp );
          
          img.FetchPatch( i, j, feat );

          auto re = std::move( pull( feat ) );

          for ( auto& ele : re ) {
            img.FetchPatch( ele.y, ele.x, feat_c );
            ranker.add( dist_l2( feat, feat_c, dim ),
                        Geometric::diff( PatLoc( -1, i, j, 0.0, 1.0 ),
                                         PatLoc( ele.id, ele.y, ele.x, 0.0, 1.0 ) ) );
          }

          
          for ( int k=0; k<ranker.len; k++ ) {
            geomap[count].push_back( ranker[k] );
          }
          
          count++;
          if ( 0 == count % 1000 ) {
            progress( count, area, "Initial Query" );
          }
        }
      }
      printf( "\n" );
      Done( "Query" );

      
      for ( int iter=0; iter<maxIter; iter++ ) {
        
        if ( 0 == ( iter & 1 ) ) {
          for ( int i=0; i<geomap.rows; i++ ) {
            for ( int j=0; j<geomap.cols; j++ ) {
              img.FetchPatch( i, j, feat );
              ranker.resize( numProp );


              // Result from last iteration
              for ( auto& ele : geomap(i,j) ) {
                PatLoc loc = ele.apply( i, j );
                if ( 0 <= loc.y && loc.y <= img.rows && 0 <= loc.x && loc.x < img.cols ) {
                  img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
                  double dist = dist_l2( feat, feat_c, img.GetPatchDim() );
                  Geometric copy = ele;
                  ranker.add( dist, copy );
                }
              }

              // random search:
              // for ( auto& ele : geomap(i,j) ) {
              //   PatLoc loc = ele.apply( i, j );
              //   Sampler4D sampler( album(loc.id).rows, album(loc.id).cols );
              //   for ( int s=0; s<numSample; s++ ) {
              //     PatLoc hypo = sampler.sample( loc );
              //     if ( 0 <= hypo.y && hypo.y <= album(loc.id).rows && 0 <= hypo.x && hypo.x < album(loc.id).cols ) {
              //       album(loc.id).FetchPatch( hypo.y, hypo.x, hypo.rotation, hypo.scale, feat_c );
              //       double dist = dist_l2( feat, feat_c, album(loc.id).GetPatchDim() );
              //       ranker.add( dist, Geometric::diff( PatLoc( -1, i, j, 0.0, 1.0 ), hypo ) );
              //     }
              //     sampler.shrink();
              //   }
              // }
                
              

              // left:
              if ( 0 < j ) {
                for ( auto& ele : geomap(i,j-1) ) {
                  PatLoc loc = ele.apply( i, j );
                  if ( 0 <= loc.y && loc.y <= img.rows && 0 <= loc.x && loc.x < img.cols ) {
                    img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
                    double dist = dist_l2( feat, feat_c, img.GetPatchDim() );
                    Geometric copy = ele;
                    ranker.add( dist, copy );
                  }
                }
              }

              // top:
              if ( 0 < i ) {
                for ( auto& ele : geomap(i-1,j ) ) {
                  PatLoc loc = ele.apply( i, j );
                  if ( 0 <= loc.y && loc.y <= img.rows && 0 <= loc.x && loc.x < img.cols ) {
                    img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
                    double dist = dist_l2( feat, feat_c, img.GetPatchDim() );
                    Geometric copy = ele;
                    ranker.add( dist, copy );
                  }
                }
              }

              
              
              geomap[ i * geomap.cols + j ].clear();
              for ( int k=0; k<ranker.len; k++ ) {
                geomap[i * geomap.cols + j ].push_back( ranker[k] );
              }
              
            }
          }
        } else {
          for ( int i=geomap.rows-1; i>=0; i-- ){
            for ( int j=geomap.cols-1; j>=0; j-- ) {
              img.FetchPatch( i, j, feat );
              ranker.resize( numProp );
              for ( auto& ele : geomap(i,j) ) {
                PatLoc loc = ele.apply( i, j );
                if ( 0 <= loc.y && loc.y <= img.rows && 0 <= loc.x && loc.x < img.cols ) {
                  img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
                  double dist = dist_l2( feat, feat_c, img.GetPatchDim() );
                  Geometric copy = ele;
                  ranker.add( dist, copy );
                }
              }


              // random search:
              // for ( auto& ele : geomap(i,j) ) {
              //   PatLoc loc = ele.apply( i, j );
              //   Sampler4D sampler( album(loc.id).rows, album(loc.id).cols );
              //   for ( int s=0; s<numSample; s++ ) {
              //     PatLoc hypo = sampler.sample( loc );
              //     if ( 0 <= hypo.y && hypo.y <= album(loc.id).rows && 0 <= hypo.x && hypo.x < album(loc.id).cols ) {
              //       album(loc.id).FetchPatch( hypo.y, hypo.x, hypo.rotation, hypo.scale, feat_c );
              //       double dist = dist_l2( feat, feat_c, album(loc.id).GetPatchDim() );
              //       ranker.add( dist, Geometric::diff( PatLoc( -1, i, j, 0.0, 1.0 ),
              //                                          hypo ) );
              //     }
              //     sampler.shrink();
              //   }
              // }
                


              // right:
              if ( j < geomap.cols - 1 ) {
                for ( auto& ele : geomap(i,j+1) ) {
                  PatLoc loc = ele.apply( i, j );
                  if ( 0 <= loc.y && loc.y <= img.rows && 0 <= loc.x && loc.x < img.cols ) {
                    img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
                    double dist = dist_l2( feat, feat_c, img.GetPatchDim() );
                    Geometric copy = ele;
                    ranker.add( dist, copy );
                  }
                }
              }

              // bottom:
              if ( i < geomap.rows - 1 ) {
                for ( auto& ele : geomap(i+1,j) ) {
                  PatLoc loc = ele.apply( i, j );
                  if ( 0 <= loc.y && loc.y <= img.rows && 0 <= loc.x && loc.x < img.cols ) {
                    img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
                    double dist = dist_l2( feat, feat_c, img.GetPatchDim() );
                    Geometric copy = ele;
                    ranker.add( dist, copy );
                  }
                }
              }

              geomap[ i * geomap.cols + j ].clear();
              for ( int k=0; k<ranker.len; k++ ) {
                geomap[i * geomap.cols + j ].push_back( ranker[k] );
              }
            }
          }
        }

        progress( iter + 1, maxIter, "PatchMatch" );
      }
      printf( "\n" );
      Done( "PatchMatch" );
      return geomap;
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
        ids.push_back( nodes[ele->query( p )].leafID );
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

    inline void FilterWeights( int thresh = 0 )
    {
      for ( int i=0; i<centers(); i++ ) {
        for ( std::unordered_map<int,int>::const_iterator iter = weights[i].begin();
              iter != weights[i].end(); ) {
          if ( iter->second <= thresh ) {
            weights[i].erase(iter++);
          } else {
            iter++;
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

    inline int totalNodes() const
    {
      return static_cast<int>( nodes.size() );
    }

    inline int maxDepth() const
    {
      int depth = 0;
      for ( auto& tree : trees ) {
        int d = tree.maxDepth();
        if ( d > depth ) {
          depth = d;
        }
      }
      return depth;
    }

  };
  
}

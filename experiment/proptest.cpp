#include <vector>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/time.hpp"
#include "LLPack/utils/pathname.hpp"
#include "LLPack/algorithms/heap.hpp"
#include "../data/FeatImage.hpp"
#include "../data/vector.hpp"
#include "../interfaces/opencv_aux.hpp"
#include "../query/tree.hpp"
#include "../graph/Trans.hpp"

using namespace PatTk;


int main( int argc, char **argv )
{

  srand( 345645631 );

  bool propagation = true;
  int numProp = 10;
  
  
  auto img = cvFeat<HOG>::gen( "simple/src.png.resized", 8, 0.9 );
  auto ref = cvFeat<HOG>::gen( "simple/ref.png", 0, 0.7 );

  if ( 0 == strcmp( argv[1], "train" ) ) {
    std::vector<FeatImage<float>::PatchProxy> l;

    for ( int i=0; i<ref.rows; i++ ) {
      for ( int j=0; j<ref.cols; j++ ) {
        l.push_back( ref.Spawn( i, j ) );
      }
    }

    int idx[l.size()];

    for ( int i=0, end=static_cast<int>( l.size() ); i < end; i++ ) {
      idx[i] = i;
    }

    Tree<SimpleKernel<float> > tree( l, idx, l.size() );

    tree.write( "tree.dat" );
  } else {
    Info( "Loading Tree ..." );
    auto tree = Tree<SimpleKernel<float> >::read( "tree.dat" );
    Done( "Loaded" );
    Info( "Querying ..." );

    float feat[img.GetPatchDim()];
    float feat_c[ref.GetPatchDim()];
    GeoMap geomap( img.rows, img.cols );
    
    int area = img.rows * img.cols;
    int count = 0;

    heap<double, Geometric> ranker( numProp );

                                
    for ( int i=0; i<img.rows; i++ ) {
      for ( int j=0; j<img.cols; j++ ) {
        if ( 999 == count % 1000 ) Info( "%d/%d", count+1, area );
        ranker.resize( numProp );
        for ( int k=0; k<10; k++ ) {
          float rotation = static_cast<float>( rand() ) / RAND_MAX * 2 * M_PI;
          // float rotation = 0.0;
          float scale = static_cast<float>( rand() ) / RAND_MAX * 0.4 + 0.8;
          img.FetchPatch( i, j, rotation, scale, feat );
          const std::vector<LocInfo>& re = tree->query( feat );
          int p = 0;
          for ( auto& ele : re ) {
            p++;
            ref.FetchPatch( ele.y, ele.x, feat_c );
            double dist = dist_l2( feat, feat_c, ref.GetPatchDim() );

            ranker.add( dist, Geometric::diff( PatLoc( -1, i, j, rotation, scale ),
                                               PatLoc( -1, ele.y, ele.x, 0.0, 1.0 ) ) );
            if ( p > 100 ) break;
          }
        }

        for ( int k=0; k<ranker.len; k++ ) {
          geomap[count].push_back( ranker[k] );
        }
        count++;
      }
    }
    Done( "Query" );

    if ( propagation ) {
      // to Right
      for ( int i=0; i<geomap.rows; i++ ) {
        Info( "%d/%d", i+1, geomap.rows );
        for ( int j=0; j<geomap.cols; j++ ) {
          img.FetchPatch( i, j, feat );
          ranker.resize( numProp );
          for ( auto& ele : geomap(i,j) ) {
            PatLoc loc = ele.apply( i, j );
            if ( 0 <= loc.y && loc.y <= ref.rows && 0 <= loc.x && loc.x < ref.cols ) {
              ref.FetchPatch( 0, loc.y, loc.x, loc.rotation, loc.scale, feat_c );
              double dist = dist_l2( feat, feat_c, ref.GetPatchDim() );
              Geometric copy = ele;
              ranker.add( dist, copy );
            }
          }

          geomap[ i * geomap.cols + j ].clear();
          for ( int k=0; k<ranker.len; k++ ) {
            geomap[ i * geomap.cols + j ].push_back( ranker[k] );
          }
          
          if ( j + 1 < geomap.cols ) {
            for ( int k=0; k<ranker.len; k++ ) {
              geomap[ i * geomap.cols + (j+1) ].push_back( ranker[k] );
            }
          }
        }
      }


      // to Bottom
      for ( int j=0; j<geomap.cols; j++ ) {
        Info( "%d/%d", j+1, geomap.cols );
        for ( int i=0; i<geomap.rows; i++ ) {
          img.FetchPatch( i, j, feat );
          ranker.resize( numProp );
          for ( auto& ele : geomap(i,j) ) {
            PatLoc loc = ele.apply( i, j );
            if ( 0 <= loc.y && loc.y <= ref.rows && 0 <= loc.x && loc.x < ref.cols ) {
              ref.FetchPatch( 0, loc.y, loc.x, loc.rotation, loc.scale, feat_c );
              double dist = dist_l2( feat, feat_c, ref.GetPatchDim() );
              Geometric copy = ele;
              ranker.add( dist, copy );
            }
          }

          geomap[ i * geomap.cols + j ].clear();
          for ( int k=0; k<ranker.len; k++ ) {
            geomap[ i * geomap.cols + j ].push_back( ranker[k] );
          }

          if ( i + 1 < geomap.rows ) {
            for ( int k=0; k<ranker.len; k++ ) {
              geomap[ (i+1) * geomap.cols + j ].push_back( ranker[k] );
            }
          }
        }
      }

      // to Left
      for ( int i=0; i<geomap.rows; i++ ) {
        Info( "%d/%d", i+1, geomap.rows );
        for ( int j=geomap.cols-1; j>=0; j-- ) {
          img.FetchPatch( i, j, feat );
          ranker.resize( numProp );
          for ( auto& ele : geomap(i,j) ) {
            PatLoc loc = ele.apply( i, j );
            if ( 0 <= loc.y && loc.y <= ref.rows && 0 <= loc.x && loc.x < ref.cols ) {                          
              ref.FetchPatch( 0, loc.y, loc.x, loc.rotation, loc.scale, feat_c );
              double dist = dist_l2( feat, feat_c, ref.GetPatchDim() );
              Geometric copy = ele;
              ranker.add( dist, copy );
            }
          }

          geomap[ i * geomap.cols + j ].clear();
          for ( int k=0; k<ranker.len; k++ ) {
            geomap[ i * geomap.cols + j ].push_back( ranker[k] );
          }

          if ( j >= 1 ) {
            for ( int k=0; k<ranker.len; k++ ) {
              geomap[ i * geomap.cols + (j-1) ].push_back( ranker[k] );
            }
          }
        }
      }


      // to Top
      for ( int j=0; j<geomap.cols; j++ ) {
        Info( "%d/%d", j+1, geomap.cols );
        for ( int i=geomap.rows-1; i>=0; i-- ) {
          img.FetchPatch( i, j, feat );
          ranker.resize( numProp );
          for ( auto& ele : geomap(i,j) ) {
            PatLoc loc = ele.apply( i, j );
            if ( 0 <= loc.y && loc.y <= ref.rows && 0 <= loc.x && loc.x < ref.cols ) {                          
              ref.FetchPatch( 0, loc.y, loc.x, loc.rotation, loc.scale, feat_c );
              double dist = dist_l2( feat, feat_c, ref.GetPatchDim() );
              Geometric copy = ele;
              ranker.add( dist, copy );
            }
          }

          geomap[ i * geomap.cols + j ].clear();
          geomap[ i * geomap.cols + j ].push_back( ranker[0] );


          if ( i >= 1 ) {
            for ( int k=0; k<ranker.len; k++ ) {
              geomap[ (i-1) * geomap.cols + j ].push_back( ranker[k] );
            }
          }
        }
      }

    }



    cv::Mat srcmat = cv::imread( "simple/src.png.resized" );
    ImageViewer srcv( "source", srcmat );
    cv::Mat refmat = cv::imread( "simple/ref.png" );
    ImageViewer refv( "reference", refmat );

    srcv.setCallback( [&geomap,&refv]( int x, int y )
                      {
                        std::vector<PatLoc> list;
                        for ( auto& ele : geomap(y,x) ) {
                          list.push_back( ele.apply( y, x ) );
                        }
                        printf( "----------------------------------------\n" );
                        for ( auto& ele : list ) {
                          ele.show();
                        }
                        refv.display( list );
                      } );

    
    
    while ( 27 != cv::waitKey(30) );
  }

  return 0;
}

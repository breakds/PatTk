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


std::vector<std::string> InitImageList()
{
  std::vector<std::string> imgList;
  imgList.push_back( "camvid_mid/Seq05VD_f00690.png" );
  imgList.push_back( "camvid_mid/Seq05VD_f00720.png" );
  imgList.push_back( "camvid_mid/Seq05VD_f00750.png" );
  imgList.push_back( "camvid_mid/Seq05VD_f00780.png" );
  imgList.push_back( "camvid_mid/Seq05VD_f00840.png" );
  imgList.push_back( "camvid_mid/Seq05VD_f00870.png" );
  imgList.push_back( "camvid_mid/Seq05VD_f00900.png" );
  imgList.push_back( "camvid_mid/Seq05VD_f00930.png" );
  imgList.push_back( "camvid_mid/Seq05VD_f00960.png" );
  imgList.push_back( "camvid_mid/Seq05VD_f00990.png" );
  imgList.push_back( "camvid_mid/Seq05VD_f01020.png" );
  imgList.push_back( "camvid_mid/Seq05VD_f01050.png" );
  return imgList;
}



int main( int argc, char **argv )
{

  if ( argc < 1 ) {
    Error( "Insufficient arguments." );
    exit( -1 );
  }
  
  
  srand( 345645631 );

  int numProp = 10;
  // int numSample = 10;
  
  
  // initialize the training set
  std::vector<std::string> imgList = std::move( InitImageList() );
  Album<float> album;
  for ( auto& ele : imgList ) {
    album.push( std::move( cvFeat<HOG>::gen( ele ) ) );
  }
  
  
  std::string srcname = "camvid_mid/Seq05VD_f00810.png";
  auto img = cvFeat<HOG>::gen( srcname, 8, 0.9 );
  
  if ( 0 == strcmp( argv[1], "train" ) ) {
    // Tree Training:

    std::vector<FeatImage<float>::PatchProxy> l;
    
    for ( int k=0; k<album.size(); k++ ) {
      auto& ref = album(k);
      for ( int i=7; i<ref.rows-7; i++ ) {
        for ( int j=7; j<ref.cols-7; j++ ) {
          l.push_back( ref.Spawn( i, j ) );
        }
      }
    }

    int idx[l.size()];

    for ( int i=0, end=static_cast<int>( l.size() ); i<end; i++ ) {
      idx[i] = i;
    }

    timer::tic();
    Tree<SimpleKernel<float> > tree( l, idx, l.size() );
    Done( "Tree built within %.5lf sec.", timer::utoc() );
    
    tree.write( "the.tree.dat" );

  } else if ( 0 == strcmp( argv[1], "test" ) ) {
    Info( "Loading Tree ..." );
    auto tree = Tree<SimpleKernel<float> >::read( "the.tree.dat" );
    Done( "Loaded" );
    Info( "Querying ..." );



    float feat[img.GetPatchDim()];
    float feat_c[img.GetPatchDim()];
    GeoMap geomap( img.rows, img.cols );


    int area = img.rows * img.cols;
    int count = 0;
    
    heap<double, Geometric> ranker( numProp );

    
    
    for ( int i=0; i<img.rows; i++ ) {
      for ( int j=0; j<img.cols; j++ ) {
        if ( 999 == count % 1000 ) Info( "%d/%d", count+1, area );
        ranker.resize( numProp );
        img.FetchPatch( i, j, feat );
        auto& re = tree->query( feat );
        int p = 0;
        for ( auto& ele : re ) {
          p++;
          album(ele.id).FetchPatch( ele.y, ele.x, feat_c );
          double dist = dist_l2( feat, feat_c, img.GetPatchDim() );

          ranker.add( dist, Geometric::diff( PatLoc( -1, i, j, 0.0, 1.0 ),
                                             PatLoc( ele.id, ele.y, ele.x, 0.0, 1.0 ) ) );
          if ( p > 100 ) break;
        }
        
        for ( int k=0; k<ranker.len; k++ ) {
          geomap[count].push_back( ranker[k] );
        }
        count++;
      }
    }
    Done( "Query" );


    for ( int iter=0; iter<15; iter++ ) {
      if ( 0 == ( iter & 1 ) ) {
        for ( int i=0; i<geomap.rows; i++ ) {
          for ( int j=0; j<geomap.cols; j++ ) {
            img.FetchPatch( i, j, feat );
            ranker.resize( numProp );


            // Result from last iteration
            for ( auto& ele : geomap(i,j) ) {
              PatLoc loc = ele.apply( i, j );
              if ( 0 <= loc.y && loc.y <= album(loc.id).rows && 0 <= loc.x && loc.x < album(loc.id).cols ) {
                album(loc.id).FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
                double dist = dist_l2( feat, feat_c, album(loc.id).GetPatchDim() );
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
                if ( 0 <= loc.y && loc.y <= album(loc.id).rows && 0 <= loc.x && loc.x < album(loc.id).cols ) {
                  album(loc.id).FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
                  double dist = dist_l2( feat, feat_c, album(loc.id).GetPatchDim() );
                  Geometric copy = ele;
                  ranker.add( dist, copy );
                }
              }
            }

            // top:
            if ( 0 < i ) {
              for ( auto& ele : geomap(i-1,j ) ) {
                PatLoc loc = ele.apply( i, j );
                if ( 0 <= loc.y && loc.y <= album(loc.id).rows && 0 <= loc.x && loc.x < album(loc.id).cols ) {
                  album(loc.id).FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
                  double dist = dist_l2( feat, feat_c, album(loc.id).GetPatchDim() );
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
              if ( 0 <= loc.y && loc.y <= album(loc.id).rows && 0 <= loc.x && loc.x < album(loc.id).cols ) {
                album(loc.id).FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
                double dist = dist_l2( feat, feat_c, album(loc.id).GetPatchDim() );
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
                if ( 0 <= loc.y && loc.y <= album(loc.id).rows && 0 <= loc.x && loc.x < album(loc.id).cols ) {
                  album(loc.id).FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
                  double dist = dist_l2( feat, feat_c, album(loc.id).GetPatchDim() );
                  Geometric copy = ele;
                  ranker.add( dist, copy );
                }
              }
            }

            // bottom:
            if ( i < geomap.rows - 1 ) {
              for ( auto& ele : geomap(i+1,j) ) {
                PatLoc loc = ele.apply( i, j );
                if ( 0 <= loc.y && loc.y <= album(loc.id).rows && 0 <= loc.x && loc.x < album(loc.id).cols ) {
                  album(loc.id).FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
                  double dist = dist_l2( feat, feat_c, album(loc.id).GetPatchDim() );
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
      Info( "iteration: %d", iter + 1 );
    }
    
    
    Done( "PatchMatch" );


    geomap.write( "candidates.dat" );


    cv::Mat srcmat = cv::imread( srcname );
    ImageViewer srcv( "source", srcmat );
    ImageViewer refv( "reference", srcmat );

    srcv.setCallback( [&album,&geomap,&imgList,&refv,&feat_c,&feat,&img]( int x, int y )
                      {
                        std::vector<PatLoc> list(1);
                        
                        img.FetchPatch( y, x, feat );

                        double minDist = 1000.0;
                        
                        for ( auto& ele : geomap(y,x) ) {
                          PatLoc loc = ele.apply(y,x);
                          album(loc.id).FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
                          double dist = dist_l2( feat, feat_c, img.GetPatchDim() );
                          if ( dist < minDist ) {
                            list[0] = loc;
                          }
                        }

                        
                        printf( "----------------------------------------\n" );
                        for ( auto& ele : list ) {
                          ele.show();
                        }
                        refv.SwitchImage( imgList[list[0].id] );
                        refv.display( list );
                      } );
    
    while ( 27 != cv::waitKey(30) );
  }
  // } else {

  //   GeoMap geomap( "candidates.dat" );

  //   float feat[img.GetPatchDim()];
  //   float feat_c[ref.GetPatchDim()];

  //   cv::Mat srcmat = cv::imread( "simple/src.png.true" );
  //   ImageViewer srcv( "source", srcmat );
  //   cv::Mat refmat = cv::imread( "simple/ref.png" );
  //   ImageViewer refv( "reference", refmat );

  //   srcv.setCallback( [&img,&ref,&feat,&feat_c,&geomap,&refv]( int x, int y )
  //                     {
  //                       std::vector<PatLoc> list;
  //                       for ( auto& ele : geomap(y,x) ) {
  //                         list.push_back( ele.apply( y, x ) );
  //                       }
  //                       printf( "----------------------------------------\n" );
  //                       img.FetchPatch( y, x, feat );
  //                       for ( auto& ele : geomap(y,x) ) {
  //                         PatLoc loc = ele.apply(y,x);
  //                         ref.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );

  //                         for ( int i=0; i<ref.GetPatchDim(); i++ ) {
  //                           if ( i % 9 == 0 ) {
  //                             printf( "--\n" );
  //                           }
  //                           printf( "%.4f\t%.4f\n", feat[i], feat_c[i] );
  //                         }
  //                       }
  //                       printf( "distance: %.4f\n", dist_l2( feat, feat_c, ref.GetPatchDim() ) );
  //                       refv.display( list );
  //                     } );

    
    
  //   while ( 27 != cv::waitKey(30) );
    
  // }

  return 0;
}
                                                                                                                                                        

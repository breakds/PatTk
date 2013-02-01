#include "LLPack/utils/Environment.hpp"
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/time.hpp"
#include "LLPack/utils/pathname.hpp"
#include "LLPack/algorithms/random.hpp"
#include "../data/FeatImage.hpp"
#include "../data/Label.hpp"
#include "../interfaces/opencv_aux.hpp"
#include "../query/forest.hpp"

using namespace EnvironmentVariable;
using namespace PatTk;
using rndgen::randperm;

int main( int argc, char **argv )
{

  if ( argc < 2 ) {
    Error( "Missing configuration file in arguments. (forestExp.conf)" );
    exit( -1 );
  }

  // srand( 7325273 );
  srand(time(NULL));

  env.parse( argv[1] );
  env.Summary();

  LabelSet::initialize( env["color-map"] );

  /* ---------- Build/Load Forest ---------- */
  std::vector<std::string> imgList = std::move( readlines( strf( "%s/%s", env["dataset"].c_str(),
                                                                 env["list-file"].c_str())) );
  auto lblList = std::move( path::FFFL( env["dataset"], imgList, "_L.png" ) );
  imgList =  std::move( path::FFFL( env["dataset"], imgList, ".png" ) );

  Album<float> album;
  {
    int i = 0;
    int n = static_cast<int>( imgList.size() );
    for ( auto& ele : imgList ) {
      album.push( std::move( cvFeat<HOG>::gen( ele ) ) );
      progress( ++i, n, "Loading Album" );
    }
  }
  printf( "\n" );

  Album<float> softAlbum;
  {
    cvFeat<SOFT_LABEL_MAP>::options.cell_side = env["lbl-size"];
    int i = 0;
    int n = static_cast<float>( lblList.size() );
    for ( auto& ele : lblList ) {
      softAlbum.push( std::move( cvFeat<SOFT_LABEL_MAP>::gen( ele ) ) );
      progress( ++i, n, "Loading Label Album" );
    }
  }
  printf( "\n" );
  softAlbum.SetPatchSize( 1 );
  softAlbum.SetPatchStride( 1 );
  

  if ( "yes" == static_cast<string>( env["build-forest"] ) ) {
    Forest<EntropyKernel<float> > forest( env["forest-size"], album, softAlbum, env["per-tree"].toDouble(),
                                         env["max-depth"] );
    forest.write( env["forest-dir"] );
  }
  
  /* ---------- Load Forest ---------- */
  Info( "Loading Forest .." );
  timer::tic();
  Forest<EntropyKernel<float> > forest( env["forest-dir"] );
  printf( "tree loaded: %.3lf sec\n", timer::utoc() );
  printf( "maxDepth: %d\n", forest.maxDepth() );

  
  /* ---------- Show Leaves ---------- */

  int viewID = randperm( album.size(), 1 )[0];
  Info( "ViewID = %d", viewID );
  
  Info( "System on." );

  cv::Mat srcmat = cv::imread( imgList[viewID] );
  cv::Mat lblmat = cv::imread( lblList[viewID] );
  ImageViewer srcview( "source", srcmat, lblmat );

  std::vector<IconList> icons;

  for ( int i=0; i<forest.size(); i++ ) {
    icons.emplace( icons.end(), strf( "tree %d", i ), 13 );
  }
  
  float feat[album(viewID).GetPatchDim()];
  
  // srcview.setCallback( [&album,&viewID,&feat,&icons,&imgList,&lblList,&forest]( int x, int y )
  //                      {
  //                        album(viewID).FetchPatch( y, x, feat );
  //                        Info( "querying (%d,%d)\n", y, x );
                         
  //                        std::vector<int> res = std::move( forest.query( feat ) );

  //                        for ( auto& item : icons ) {
  //                          item.clear();
  //                        }

  //                        int viewTrees = 4;
  //                        int i = 0;
  //                        for ( auto& leafID : randperm<int>( res, viewTrees )  ) {
  //                          int count = 0;
  //                          for ( auto& loc : forest(leafID).store ) {
  //                            if ( count++ > 200 ) break;
  //                            icons[i].push( imgList, PatLoc( loc ) );
  //                            icons[i].push( lblList, PatLoc( loc ) );
  //                          }
  //                          i++;
  //                        }

  //                        for ( int v=0; v<viewTrees; v++  ) {
  //                          icons[v].display();
  //                        }
                         
  //                      } );

  srcview.setCallback( [&album,&viewID,&feat,&icons,&imgList,&lblList,&forest]( int x, int y )
                       {
                         album(viewID).FetchPatch( y, x, feat );
                         Info( "querying (%d,%d)\n", y, x );
                         
                         std::vector<int> res = std::move( forest.query_node( feat, 18 ) );

                         icons[0].clear();
                         icons[1].clear();

                         int pick_tree = rand() % forest.size();
                         
                         auto childID = forest.getChildren( res[pick_tree] );

                         if ( -1 != childID.first ) {
                           std::vector<LocInfo> left = std::move( forest.collect( childID.first ) );
                           std::vector<LocInfo> right = std::move( forest.collect( childID.second ) );
                           printf( "left: %ld\n", left.size() );
                           printf( "right: %ld\n", right.size() );
                           int count = 0;
                           for ( auto& loc : left ) {
                             if ( count++ > 400 ) break;
                             icons[0].push( imgList, PatLoc( loc ) );
                             icons[0].push( lblList, PatLoc( loc ) );
                           }
                           count = 0;
                           for ( auto& loc : right ) {
                             if ( count++ > 400 ) break;
                             icons[1].push( imgList, PatLoc( loc ) );
                             icons[1].push( lblList, PatLoc( loc ) );
                           }
                         }
                         
                         icons[0].display();
                         icons[1].display();
                         
                       } );

  
  
  while( 27 != cv::waitKey() );

  
  
  return 0;
}



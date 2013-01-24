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

  srand( 7325273 );

  env.parse( argv[1] );
  env.Summary();

  LabelSet::initialize( env["color-map"] );

  /* ---------- Build/Load Forest ---------- */
  std::vector<std::string> imgList = std::move( readlines( strf( "%s/%s", env["dataset"].c_str(),
                                                                 env["list-file"].c_str())) );
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

  
  if ( "yes" == static_cast<string>( env["build-forest"] ) ) {
    Forest<SimpleKernel<float> > forest( env["forest-size"], album, env["per-tree"].toDouble(),
                                         env["max-depth"] );
    forest.write( env["forest-dir"] );
  }
  
  /* ---------- Load Forest ---------- */
  Info( "Loading Forest .." );
  timer::tic();
  Forest<SimpleKernel<float> > forest( env["forest-dir"] );
  printf( "tree loaded: %.3lf sec\n", timer::utoc() );
  printf( "maxDepth: %d\n", forest.maxDepth() );

  
  /* ---------- Show Leaves ---------- */

  int viewID = randperm( album.size(), 1 )[0];
  Info( "ViewID = %d", viewID );
  
  Info( "System on." );

  cv::Mat srcmat = cv::imread( imgList[viewID] );
  ImageViewer srcview( "source", srcmat );

  std::vector<IconList> icons;

  for ( int i=0; i<forest.size(); i++ ) {
    icons.emplace( icons.end(), strf( "tree %d", i ), 13 );
  }
  
  float feat[album(viewID).GetPatchDim()];

  forest.getChildren( 12 );
  
  srcview.setCallback( [&album,&viewID,&feat,&icons,&imgList,&forest]( int x, int y )
                       {
                         album(viewID).FetchPatch( y, x, feat );
                         Info( "querying (%d,%d)\n", y, x );

                         std::vector<int> res = std::move( forest.query( feat ) );

                         for ( auto& item : icons ) {
                           item.clear();
                         }

                         int i = 0;
                         for ( auto& leafID : res ) {
                           int count = 0;
                           for ( auto& loc : forest(leafID).store ) {
                             if ( count++ > 200 ) break;
                             icons[i].push( imgList, PatLoc( loc ) );
                           }
                           i++;
                         }

                         for ( auto& item : icons ) {
                           item.display();
                         }
                         
                       } );
  
  
  while( 27 != cv::waitKey() );
  
  return 0;
}



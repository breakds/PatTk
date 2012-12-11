#include <vector>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/time.hpp"
#include "LLPack/utils/pathname.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/algorithms/heap.hpp"
#include "../data/FeatImage.hpp"
#include "../data/vector.hpp"
#include "../interfaces/opencv_aux.hpp"
#include "../query/forest.hpp"
#include "../graph/Trans.hpp"

using namespace PatTk;
using namespace EnvironmentVariable;


int main( int argc, char **argv )
{

  if ( argc < 2 ) {
    Error( "Missing configuration file in options. (testing.conf)" );
    exit( -1 );
  }

  srand( 345645631 );


  env.parse( argv[1] );
  env.Summary();


  std::vector<std::string> imgList = std::move( readlines( strf( "%s/%s", env["dataset"].c_str(),
                                                                 env["list-file"].c_str())) );
  imgList = std::move( path::FFFL( env["dataset"], imgList, ".png" ) );
  
  for ( auto& ele : imgList ) {
    printf( "%s\n", ele.c_str() );
  }


  Info( "Loading Learning Album ..." );
  Album<float> album;
  for ( auto& ele : imgList ) {
    album.push( std::move( cvFeat<HOG>::gen( ele ) ) );
  }
  Done( "Learning Album Loaded" );
  
  
  Info( "Loading Forest ..." );
  Forest<SimpleKernel<float> > forest( env["forest-name"] );
  
  Info( "Learning ..." );

  forest.PrepareWeitghts();
  
  float feat[album(0).GetPatchDim()];

  for ( int k=0; k<album.size(); k++ ) {
    for ( int i=0; i<album(k).rows; i++ ) {
      for ( int j=0; j<album(k).cols; j++ ) {
        album(k).FetchPatch( i, j, feat );
        forest.learn( feat );
      }
    }
    Info( "%d / %d learned.", k + 1, album.size() );
  }
  
  forest.writeWeights( env["forest-name"] );
  
  return 0;
}
                                                                                                                                                        

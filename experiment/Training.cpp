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
    Error( "Missing configuration file in options. (training.conf)" );
    exit( -1 );
  }
  
  srand( 345645631 );

  env.parse( argv[1] );
  env.Summary();
  
  std::vector<std::string> imgList = std::move( readlines( strf( "%s/%s", env["dataset"].c_str(),
                                                                 env["list-file"].c_str())) );
  imgList =  std::move( path::FFFL( env["dataset"], imgList, ".png" ) );

  for ( auto& ele : imgList ) {
    printf( "%s\n", ele.c_str() );
  }


  Info( "Loading Training Album ..." );
  Album<float> album;
  for ( auto& ele : imgList ) {
    album.push( std::move( cvFeat<HOG>::gen( ele ) ) );
  }
  Done( "Training Album Loaded" );
  
  std::vector<FeatImage<float>::PatchProxy> l;
    
  for ( int k=0; k<album.size(); k++ ) {
    auto& ref = album(k);
    for ( int i=7; i<ref.rows-7; i++ ) {
      for ( int j=7; j<ref.cols-7; j++ ) {
        l.push_back( ref.Spawn( i, j ) );
      }
    }
  }


  timer::tic();
  Forest<SimpleKernel<float> > forest( 10, l, 0.2 );
  Done( "Tree built within %.5lf sec.", timer::utoc() );

  forest.write( env["forest-name"] );

  return 0;
}

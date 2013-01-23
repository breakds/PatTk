#include "LLPack/utils/Environment.hpp"
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/time.hpp"
#include "LLPack/utils/pathname.hpp"
#include "../data/FeatImage.hpp"
#include "../data/Label.hpp"
#include "../interfaces/opencv_aux.hpp"
#include "../query/forest.hpp"

using namespace EnvironmentVariable;
using namespace PatTk;

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
  

  timer::tic();
  Forest<SimpleKernel<float> > forest( env["forest-dir"] );
  printf( "tree loaded: %.3lf sec\n", timer::utoc() );

  printf( "maxDepth: %d\n", forest.maxDepth() );

  
  /* ---------- Show Leaves ---------- */
  for ( int i=0; i<forest.centers(); i++ ) {
    
  }
  
  
  
  
  
  return 0;
}



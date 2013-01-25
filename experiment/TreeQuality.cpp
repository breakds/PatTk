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
    Error( "Missing configuration file in arguments. (treeQ.conf)" );
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


  Album<uchar> lblAlbum;
  {
    int i = 0;
    int n = static_cast<int>( lblList.size() );
    for ( auto& ele : lblList ) {
      lblAlbum.push( std::move( cvFeat<BGR>::gen( ele ) ) );
      progress( ++i, n, "Loading Label Album" );
    }
  }
  printf( "\n" );
  lblAlbum.SetPatchSize( env["lbl-size"] );
  lblAlbum.SetPatchStride( 1 );
  

  
  /* ---------- Load Forest ---------- */
  Info( "Loading Forest .." );
  timer::tic();
  Forest<SimpleKernel<float> > forest( env["forest-dir"] );
  printf( "tree loaded: %.3lf sec\n", timer::utoc() );
  printf( "maxDepth: %d\n", forest.maxDepth() );

  
  /* ---------- Collective Entropy ---------- */
  uchar label[lblAlbum(0).GetPatchDim()];
  WITH_OPEN( out, env["entropy-output"].c_str(), "w" );
  int count[LabelSet::classes];
  for ( int i=0; i<forest.centers(); i++ ) {
    memset( count, 0, sizeof(int) * LabelSet::classes );
    for ( auto& ele : forest(i).store ) {
      lblAlbum(ele.id).FetchPatch( ele.y, ele.x, label );
      for ( int j=0; j<lblAlbum(0).GetPatchDim(); j+=3 ) {
        count[LabelSet::GetClass( label[j], label[j+1], label[j+2] )]++;
      }
    }
    double ent = entropy( count, LabelSet::classes );
    fprintf( out, "%.8lf\n", ent );
    if ( 0 == i % 100 ) progress( i+1, forest.centers(), "Calculating Entropy" );
  }
  printf( "\n" );
  END_WITH( out );
  

  
  return 0;
}



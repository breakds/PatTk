#include <vector>
#include <string>
#include <cmath>

#include "LLPack/utils/Environment.hpp"
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/pathname.hpp"
#include "../interfaces/cv_interface.hpp"
#include "../graph/update.hpp"

using namespace PatTk;

int main( int argc, char **argv )
{
  if ( argc < 2 ) {
    Error( "Missing configuration file in options. (show.conf)" );
    exit( -1 );
  }

  env.parse( argv[1] );
  env.Summary();

  // Image List
  std::vector<std::string> imgList = std::move( readlines( strf( "%s/%s", env["dataset"].c_str(),
                                                                 env["list-file"].c_str() ) ) );

  // Album
  auto album = std::move( cvAlbumGen<BGRCell,int,false>::gen( path::FFFL( env["dataset"], imgList, ".png" ) ) );
  
  album.SetPatchParameter( env["patch-w"], env["patch-w"] );
  
  // target
  int tar = 0;

  int ref = 4;

  std::vector<AffineTransform> trans;
  WITH_OPEN( in, env["transform"].c_str(), "r" );
  int K = 0;
  fread( &K, sizeof(int), 1, in );
  trans.resize( K );
  for ( int k=0; k<K; k++ ) {
    trans[k].read(in);
  }
  END_WITH( in );

  PatGraph graph( album(tar).rows, album(tar).cols );

  int patchSide = env["patch-w"];

  for ( int i=0; i<album(tar).rows; i++ ) {
    Info( "%d/%d", i+1, album(tar).rows );
    for ( int j=0; j<album(tar).cols; j++ ) {
      auto p0 = album(tar).Spawn( i, j, 1.0, 0.0 );

      for ( auto& ele : trans ) {
        PatLoc tmp( ele, ref, i, j );
        if ( 0 <= tmp.y && tmp.y <album(ref).rows - patchSide &&
             0 <= tmp.x && tmp.x <album(ref).cols - patchSide ) {
          int sum = 0;
          auto p1 = tmp.toPatch( album );
          for ( int c=0, end=p0.dim(); c<end; c++ ) {
            sum += (p1[c] - p0[c]) * (p1[c] - p0[c]);
          }
          tmp.dist = static_cast<float>( sum );
        } else {
          tmp.y = 10000;
          tmp.x = 10000;
          tmp.index = -1;
          tmp.dist = 48000000000.0f;
          tmp.scale = 100.0;
          tmp.rotation = 0.0;
        }
        graph[i * album(tar).cols + j].push_back( tmp );
      }

    }
  }

  Done( "Graph Generated." );

  std::string filename = "mapping.dat";
  graph.write( filename );

  
  UpdateGraph( imgList, album, album(tar).rows, album(tar).cols, tar, ref );

  
  return 0;
}

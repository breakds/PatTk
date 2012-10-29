/*********************************************************************************
 * File: Clustering.hpp
 * Description: Clustering of nearest neighbor offsets
 * by BreakDS, @ University of Wisconsin-Madison, Thu Oct 18 15:36:25 CDT 2012
 *********************************************************************************/

#include "LLPack/utils/pathname.hpp"
#include "../interfaces/cv_interface.hpp"
#include "../graph/Graph.hpp"
#include "../data/metric.hpp"


using namespace EnvironmentVariable;
using namespace PatTk;

int main( int argc, char **argv )
{
  if ( argc < 2 ) {
    Error( "Missing configuration file in options." );
    exit( -1 );
  }

  env.parse( argv[1] );
  env.Summary();

  std::vector<std::string> imgList = std::move( readlines( strf( "%s/%s", env["dataset"].c_str(),
                                                                 env["list-file"].c_str() ) ) );

  int side = env["patch-w"];
  auto album = std::move( cvAlbumGen<BGRCell,int,false>::gen( path::FFFL( env["dataset"], imgList, ".png" ) ) );
  album.SetPatchParameter( side, side );
  
  PatGraph graph( env["graph-file"] );
  PatGraph final( graph.rows, graph.cols );

  int tar = env["original"];
  
  for ( int i=0; i<graph.rows-side; i++ ) {
    for ( int j=0; j<graph.cols-side; j++ ) {

      auto p0 = album(tar).Spawn( i, j, 1.0, 0.0 );

      int min = 0;
      int minp = -1;

      for ( int k=0; k<static_cast<int>( graph(i,j).size() ); k++ ) {
        int y = static_cast<int>( graph(i,j)[k].y );
        int x = static_cast<int>( graph(i,j)[k].x );
        if ( y < album(graph(i,j)[k].index).rows - side &&
             x < album(graph(i,j)[k].index).cols - side ) {
          auto p1 = graph(i,j)[k].toPatch( album );
          int dist = L1Dist<BGRCell,int,false>( p0, p1 );
          if ( -1 == minp || dist < min ) {
            min = dist;
            minp = k;
          }
        }
        /*
        else {
          printf( "(%d,%d) -> (%d,%d)\n", i, j, y, x );
        }
        */
      }
      if ( -1 == minp ) {
        Error( "corrupted candidates for (%d,%d)", i, j );
        exit( -1 );
      }
      final[i*graph.cols+j].push_back( graph(i,j)[minp] );
    }
  }

  for ( int i=0; i<graph.rows; i++ ) {
    for ( int j=0; j<graph.cols; j++ ) {
      if ( 0 == final[i*graph.cols+j].size() ) {
        final[i*graph.cols+j].push_back( graph(i,j)[0] );
      }
    }
  }
  

  

  std::string savepath = env["output"];
  final.write( savepath );

  return 0;

  
}

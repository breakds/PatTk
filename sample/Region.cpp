/*********************************************************************************
 * File: Region.cpp
 * Description: Find common region in images
 * An implementation of Faktor's ECCV'12 paper
 * "Clustering by Composition"
 * by BreakDS, @ University of Wisconsin-Madison, Tue Oct 16 16:25:46 CDT 2012
 *********************************************************************************/

#include <string>
#include <vector>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/utils/pathname.hpp"
#include "../data/metric.hpp"
#include "../interfaces/cv_interface.hpp"




using namespace path;
using namespace PatTk;
using namespace EnvironmentVariable;

using std::vector;




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


  int tar = env["target"];
  int src = env["source"];

  
  auto album = std::move( cvAlbumGen<LabCell,int,true>::gen( FFFL( env["dataset"], imgList, ".png" ) ) );
  album.SetPatchParameter( env["patch-w"], env["patch-w"] );


  // Sampling
  int level = env["sample-level"];
  int side = env["patch-w"];
  PatGraph graph( album(tar).rows, album(tar).cols );
  for ( int i=0; i<album(tar).rows-side; i++ ) {
    for ( int j=0; j<album(tar).cols-side; j++ ) {
      graph[i*album(tar).cols+j].reserve( level + 10 );
      for ( int k=0; k<level; k++ ) {
        int y = rand() % (album(src).rows-side);
        int x = rand() % (album(src).cols-side);
        graph[i*album(tar).cols+j].push_back( PatLoc( src, y, x, 1.0, 0.0, 0.0 ) );
      }
    }
  }

  // Sweeping Pass 1
  for ( int i=0; i<album(tar).rows-side; i++ ) {
    for ( int j=0; j<album(tar).cols-side; j++ ) {
      int min = -1;
      PatLoc selected( 0, 0, 0, 1.0, 0, 0 );
      for ( auto& ele : graph(i,j) ) {
        int dist = L1Dist( album(tar)(i,j), album(src)(ele.y,ele.x) );
        if ( -1 == min || dist < min ) {
          min = dist;
          selected = ele;
        }
      }
      graph[i*album(tar).cols+j].clear();
      graph[i*album(tar).cols+j].reserve(10);
      graph[i*album(tar).cols+j].push_back( selected );

      // propagate downward
      if ( i+1<album(tar).rows-side && selected.y + 1 < album(src).rows-side ) {
        selected.y++;
        graph[(i+1) * album(tar).cols + j].push_back( selected );
        selected.y--;
      }

      // propagate to right
      if ( j+1<album(tar).cols-side && selected.x + 1 < album(src).cols-side ) {
        selected.x++;
        graph[i * album(tar).cols + j + 1].push_back( selected );
        selected.x--;
      }
    }
  }

  // Sweeping Pass 2
  for ( int i=album(tar).rows-side; i>=0; i-- ) {
    for ( int j=album(tar).cols-side; j>=0; j-- ) {
      int min = -1;
      PatLoc selected( 0, 0, 0, 1.0, 0, 0 );
      for ( auto& ele : graph(i,j) ) {
        int dist = L1Dist( album(tar)(i,j), album(src)(ele.y,ele.x) );
        if ( -1 == min || dist < min ) {
          min = dist;
          selected = ele;
        }
      }
      graph[i*album(tar).cols+j].clear();
      graph[i*album(tar).cols+j].reserve(10);
      graph[i*album(tar).cols+j].push_back( selected );

      // propagate upward
      if ( i-1 >= 0 && selected.y - 1 >= 0 ) {
        selected.y--;
        graph[(i-1) * album(tar).cols + j].push_back( selected );
        selected.y++;
      }

      // propagate to right
      if ( j-1 >=0 && selected.x - 1 >= 0 ) {
        selected.x--;
        graph[i * album(tar).cols + j - 1].push_back( selected );
        selected.x++;
      }
    }
  }

  // finalize
  for ( int i=0; i<album(tar).rows * album(tar).cols; i++ ) {
    if ( graph(i).empty() ) {
      graph[i].push_back( PatLoc( src, 0, 0, 1.0, 0, 0 ) );
    }
  }

  std::string savepath = strf( "%s/%s.graph", env["graph-dir"].c_str(), imgList[tar].c_str() );
  graph.write( savepath );


  return 0;
}

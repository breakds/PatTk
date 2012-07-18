/*********************************************************************************
 * File: AlbumTest.cpp
 * Description: Sample program for using 2d::Album.
 * by BreakDS, University of Wisconsin Madison, Fri Jul 13 20:49:44 CDT 2012
 *********************************************************************************/

#include <utility>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/utils/pathname.hpp"
#include "data/features.hpp"
#include "opencv2/opencv.hpp"
#include "interfaces/cv_interface.hpp"


using namespace PatTk;
using namespace EnvironmentVariable;

int main( int argc, char **argv )
{

  if ( argc < 2 ) {
    Error( "Not enough arguments." );
    exit( -1 );
  }

  env.parse( argv[1] );
  env.Summary();
  

  
  // path::FFFL will create a filelist (vector<string>)
  // pass it to cvAlbumGen::gen will generate an album
  // using std::move to avoid lvalue copy
  auto album = std::move( cvAlbumGen<LabCell,int>::gen( path::FFFL( env["folder"], env["files"], ".png" ) ) );



  for ( int i=0; i<album.size(); i++ ) {
    // print image id
    printf( "id: %d, ", album(i).id );
    // Summary each individual image
    album(i).Summary();
  }
  
  return 0;
}



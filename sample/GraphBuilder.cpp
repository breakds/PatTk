/*********************************************************************************
 * File: GraphBuilder.hpp
 * Description: Build NN Graph for patches
 * by BreakDS, @ University of Wisconsin-Madison, Sun Aug 19 08:49:57 CDT 2012
 *********************************************************************************/

#include <vector>
#include <string>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/utils/pathname.hpp"
#include "../interfaces/cv_interface.hpp"
#include "../graph/update.hpp"

using namespace PatTk;
using namespace EnvironmentVariable;

int main( int argc, char **argv )
{
  if ( argc < 2 ) {
    Error( "Missing configuration file in options." );
    exit( -1 );
  }
  env.parse( argv[1] );
  env.Summary();
  
  std::vector<std::string> imgList = std::move( path::FFFL( "", env["files"], "" ) );
  auto album = std::move( cvAlbumGen<HoGCell,int,false>::gen( path::FFFL( env["folder"], env["files"], ".png" ) ) );
  
  UpdateGraph( imgList, album, 1, 0 );
  return 0;
                          
}

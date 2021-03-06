/*********************************************************************************
 * File: GraphBuilder.cpp
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


using namespace path;
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

  
  
  std::vector<std::string> imgList = std::move( readlines( strf( "%s/%s", env["dataset"].c_str(),
                                                                 env["list-file"].c_str() ) ) );

  Album<BGRCell,int,false> album;
  if ( 1 < env["enrich-iter"] ) {
    Info( "Loading dataset ..." );
    album = std::move( cvAlbumGen<BGRCell,int,false>::gen( FFFL( env["dataset"], imgList, ".png" ) ) );
    album.SetPatchParameter( env["patch-w"], env["patch-w"] );
    Done( "Dataset loaded" );
  }

  int tar = 0;
  
  int tarH = 0;
  int tarW = 0;
  {
    cv::Mat img = cv::imread( strf( "%s/%s.png", env["dataset"].c_str(), imgList[tar].c_str() ) );
    if ( img.empty() ) {
      Error( "Cannot open %s.", strf( "%s/%s.png", env["dataset"].c_str(), imgList[tar].c_str() ).c_str() );
      exit( -1 );
    }
    tarH = img.rows;
    tarW = img.cols;
  }

  for ( int ref=4; ref<5; ref++ ) {
    double energy = UpdateGraph( imgList, album, tarH, tarW, tar, ref );
    WITH_OPEN( out, "status.txt", "a" );
    fprintf( out, "%d: %.4lf\n", ref, energy );
    END_WITH( out );
  }
  

  

  return 0;
}

/*********************************************************************************
 * File: CandView.hpp
 * Description: Candidate Viewer
 * by BreakDS, @ University of Wisconsin-Madison, Fri Aug 24 14:24:49 CDT 2012
 *********************************************************************************/

#include "LLPack/utils/Environment.hpp"

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
  string path = strf( "%s/%s.graph", env["graph-dir"].c_str(), imgList[394].c_str() );
  PatGraph graph( path );
  

}

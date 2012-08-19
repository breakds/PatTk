/*********************************************************************************
 * File: update.hpp
 * Description: update the candiadte set of a certain image
 * by BreakDS, University of Wisconsin Madison, Sat Aug 18 16:24:17 CDT 2012
 *********************************************************************************/

#pragma once
#include <cassert>
#include <string>
#include "LLPack/utils/Environment.hpp"
#include "Graph.hpp"

using namespace EnvironmentVariable;

namespace PatTk
{
  namespace
  {
    GenConfDefault( const std::string &directory, const std::string &target,
                   const std::string &reference )
    {
      WITH_OPEN( out, "PatchMatch.conf", "w" );
      fprintf( out, "(include pm_common.conf)\n" );
      fprintf( out, "(directory %s)\n", directory.c_str() );
      fprintf( out, "(target %s)\n", target.c_str() );
      fprintf( out, "(files %s)\n", target.c_str() );
      fprintf( out, "(algorithm rotscale)\n" );
      fprintf( out, "(patch-w 17)\n" );
      fprintf( out, "(nn-iters -1)\n" );
      fprintf( out, "(rs-max -1)\n" );
      fprintf( out, "(rs-min -1)\n" );
      fprintf( out, "(rs-ratio -1)\n" );
      fprintf( out, "(rs-iters -1)\n" );
      fprintf( out, "(cores 6)\n" );
      fprintf( out, "(bmask null)\n" );
      fprintf( out, "(window-h -1)\n" );
      fprintf( out, "(window-w -1)\n" );
      fprintf( out, "(ann-prev null)\n" );
      fprintf( out, "(ann-window null)\n" );
      fprintf( out, "(ann-winsize null)\n" );
      fprintf( out, "(scalerange 2)\n" );
      END_WITH( out );
    }
  }
  
  void UpdateGraph( const std::string &directory,
                    const std::vector<std::string> &imgList, // filenames for all the images
                    int targetID, // image id of the target
                    int referenceID ) // image id of the reference
  {
    // Generate the configuration file
    ::GenConfDefault( directory, imgList[targetID], imgList[referenceID] );

    // Call nnmex externally
    system( "nnmex PatchMatch.conf" );
    
    // TODO: PatGraph Loading Stuff

    
    
  }
};

/*********************************************************************************
 * File: Stiching.hpp
 * Description: Unit Test for update.hpp
 * by BreakDS, University of Wisconsin Madison, Mon Aug 27 15:59:06 CDT 2012
 *********************************************************************************/


#include "opencv2/opencv.hpp"
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "../interfaces/cv_interface.hpp"
#include "../graph/update.hpp"
#include <string>
#include <vector>
#include <cmath>

using namespace EnvironmentVariable;
using namespace PatTk;


void drawBox( cv::Mat& canvas, const PatLoc& loc, int side )
{
  float dy = sin( loc.rotation ) * side;
  float dx = cos( loc.rotation ) * side;
  
  cv::line( canvas,
            cv::Point( loc.x, loc.y ),
            cv::Point( loc.x + dx, loc.y - dy ),
            cv::Scalar( 0, 255, 0 ) );

  cv::line( canvas,
            cv::Point( loc.x + dx, loc.y - dy),
            cv::Point( loc.x + dx + dy, loc.y - dy + dx ),
            cv::Scalar( 0, 255, 0 ) );

  cv::line( canvas,
            cv::Point( loc.x + dx + dy, loc.y - dy + dx ),
            cv::Point( loc.x + dy, loc.y + dx ),
            cv::Scalar( 0, 255, 0 ) );

  cv::line( canvas,
            cv::Point( loc.x + dy, loc.y + dx ),
            cv::Point( loc.x, loc.y ),
            cv::Scalar( 0, 255, 0 ) );
}



int main( int argc, char **argv )
{
  InitializeEnvironment( argc, argv, true );
  
  std::vector<std::string> imgList = std::move( readlines( strf( "%s/%s", env["dataset"].c_str(),
                                                                 env["list-file"].c_str() ) ) );
  int tar = env["target"];
  int ref = env["source"];


  cv::Mat mat = cv::imread( strf( "%s/%s.png", env["dataset"].c_str(), imgList[tar].c_str() ) );
  if ( mat.empty() ) {
    Error( "Cannot open %s.", strf( "%s/%s.png", env["dataset"].c_str(), imgList[tar].c_str() ).c_str() );
    exit( -1 );
  }
  int tarH = mat.rows;
  int tarW = mat.cols;

  
  UpdateGraph( imgList, tarH, tarW, tar, ref );
  string path = strf( "%s/%s.graph", env["graph-dir"].c_str(), imgList[tar].c_str() );
  PatGraph graph( path );


  // Interactive Interfaces:
  cv::Mat canvas = mat.clone();
  cv::imshow( "show", canvas );
  int curX = 0;
  int curY = 0;
  char key;
  IconList<PatLoc> list( "patches" );
  list.options.zoom = 3;

  
  // Press ESC to terminate
  while ( 27 != ( key = cv::waitKey(30) ) ) {

    bool changed = false;

    // Smaller Step
    if ( 97 == key ) {
      // Left
      if ( curX > 0 ) {
        curX--;
      }
      changed = true;
    } else if ( 100 == key ) {
      // Right
      if ( curX < mat.cols - 1) {
        curX++;
      }
      changed = true;
    } else if (  115 == key ) {
      // Down
      if ( curY < mat.rows - 1 ) {
        curY++;
      }
      changed = true;
    } else if ( 119 == key ) {
      // Up
      if ( curY > 0 ) {
        curY--;
      }
      changed = true;
    }
    
    // Larger Step
    if ( 65 == key ) {
      // Left
      if ( curX - 10 >= 0 ) {
        curX-=10;
      }
      changed = true;
    } else if ( 68 == key ) {
      // Right
      if ( curX + 10 < mat.cols ) {
        curX+=10;
      }
      changed = true;
    } else if (  83 == key ) {
      // Down
      if ( curY + 10 < mat.rows ) {
        curY+=10;
      }
      changed = true;
    } else if ( 87 == key ) {
      // Up
      if ( curY - 10 >= 0 ) {
        curY-=10;
      }
      changed = true;
    }
    
    if ( changed ) {
      canvas = mat.clone();
      PatLoc loc( tar, curY, curX, 1.0, 0.0, 0.0 );
      drawBox( canvas, loc, env["patch-side"] );
      cv::imshow( "show", canvas );
            
      list.clear();
      list.push( imgList, loc, env["patch-side"], env["patch-side"] );
      for ( auto& ele : graph( curY, curX ) ) {
        list.push( imgList, ele, env["patch-side"], env["patch-side"] );
      }
      list.setCallback( [&]( const int index ){ printf( "target:(%d,%d) - ", curY, curX );
          graph(curY,curX)[index-1].show(); } );
      list.display();
    }
  }


  return 0;
}

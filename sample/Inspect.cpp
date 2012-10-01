/*********************************************************************************
 * File: Inspect.hpp
 * Description: Validation of optimization result
 * by BreakDS, @ University of Wisconsin-Madison, Mon Oct  1 08:19:04 CDT 2012
 *********************************************************************************/

#include <vector>
#include <string>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/algorithms/sort.hpp"
#include "../interfaces/cv_interface.hpp"
#include "../graph/Graph.hpp"

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
  
  PatGraph graph( env["graph-file"] );

  cv::Mat idxImg( graph.rows, graph.cols, CV_8UC3 );
  cv::Mat locImg( graph.rows, graph.cols, CV_8UC3 );
  cv::Mat sclImg( graph.rows, graph.cols, CV_8UC3 );
  cv::Mat rotImg( graph.rows, graph.cols, CV_8UC3 );

  for ( int i=0; i<graph.rows; i++ ) {
    for ( int j=0; j<graph.cols; j++ ) {

      // fill index image;
      uchar b_channel = graph(i,j)[0].index % 255;
      uchar g_channel = graph(i,j)[0].index / 255;
      idxImg.at<cv::Vec3b>( i, j )[0] = b_channel;
      idxImg.at<cv::Vec3b>( i, j )[1] = g_channel;
      idxImg.at<cv::Vec3b>( i, j )[2] = 0;

      // fill location image;
      b_channel = static_cast<uchar>( graph(i,j)[0].y / static_cast<double>( graph.rows ) * 255.0 );
      g_channel = static_cast<uchar>( graph(i,j)[0].x / static_cast<double>( graph.cols ) * 255.0 );
      locImg.at<cv::Vec3b>( i, j )[0] = b_channel;
      locImg.at<cv::Vec3b>( i, j )[1] = g_channel;
      locImg.at<cv::Vec3b>( i, j )[2] = 0;


      // fill scale image;
      b_channel = static_cast<uchar>( graph(i,j)[0].scale / 10.0 * 255.0 );
      sclImg.at<cv::Vec3b>( i, j )[0] = b_channel;
      sclImg.at<cv::Vec3b>( i, j )[1] = b_channel;
      sclImg.at<cv::Vec3b>( i, j )[2] = b_channel;

      // fill rotation image;
      b_channel = static_cast<uchar>( ( sin( graph(i,j)[0].rotation ) + 1.0 ) / 2.0 * 255.0 );
      g_channel = static_cast<uchar>( ( cos( graph(i,j)[0].rotation ) + 1.0 ) / 2.0 * 255.0 );
      rotImg.at<cv::Vec3b>( i, j )[0] = b_channel;
      rotImg.at<cv::Vec3b>( i, j )[1] = g_channel;
      rotImg.at<cv::Vec3b>( i, j )[2] = 0;
      
    }
  }

  imshow( "Index", idxImg );

  imshow( "Scale", sclImg );

  imshow( "Location", locImg );

  imshow( "Rotation", rotImg );
  
  while ( cv::waitKey(30) != 27 );
  
  return 0;
  
}

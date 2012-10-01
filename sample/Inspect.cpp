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

cv::Mat vappend( const cv::Mat img0, cv::Mat img1 )
{
  if ( img0.cols != img1.cols ) {
    Error( "vappend(): dimension mismatch." );
    exit( -1 );
  }
  cv::Mat result( img0.rows + img1.rows, img0.cols, CV_8UC3 );
  img0.copyTo( result( cv::Rect( 0, 0, img0.cols, img0.rows ) ) );
  img1.copyTo( result( cv::Rect( 0, img0.rows, img1.cols, img1.rows ) ) );
  return result;
}

cv::Mat happend( const cv::Mat img0, cv::Mat img1 )
{
  if ( img0.rows != img1.rows ) {
    Error( "happend(): dimension mismatch." );
    exit( -1 );
  }
  cv::Mat result( img0.rows, img0.cols + img1.cols, CV_8UC3 );
  img0.copyTo( result( cv::Rect( 0, 0, img0.cols, img0.rows ) ) );
  img1.copyTo( result( cv::Rect( img0.cols, 0, img1.cols, img1.rows ) ) );
  return result;
}


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
      uchar b_channel = graph(i,j)[0].index * 10 % 255;
      uchar g_channel = graph(i,j)[0].index * 10 / 255;
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


  /*
   * | index | location |
   * | scale | rotation |
   */
  cv::Mat result = happend( vappend( idxImg, sclImg ), vappend( locImg, rotImg ) );

  cv::imshow( "Result", result );

  cv::imwrite( env["output"], result );
  
  while ( cv::waitKey(30) != 27 );

  
  
  return 0;
  
}
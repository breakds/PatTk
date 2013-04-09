#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "LLPack/utils/extio.hpp"
#include "../interfaces/opencv_aux.hpp"


using namespace PatTk;

int main( int __attribute__((__unused__)) argc, char** argv )
{

  srand( time( NULL ) );
  auto img = cvFeat<LabF>::gen( argv[1] );
  cv::Mat raw = cv::imread( argv[1] );
  cv::Mat lab;
  cv::cvtColor( raw, lab, CV_BGR2Lab );


  for ( int i=0; i<raw.rows; i++ ) {
    for ( int j=0; j<raw.rows; j++ ) {
      if ( rand() % 100 < 20 ) {
        DebugInfo( "%hhu %hhu %hhu",
                   lab.at<cv::Vec3b>( i, j )[0],
                   lab.at<cv::Vec3b>( i, j )[1],
                   lab.at<cv::Vec3b>( i, j )[2] );
        DebugInfo( "%hhu %hhu %hhu",
                   raw.at<cv::Vec3b>( i, j )[0],
                   raw.at<cv::Vec3b>( i, j )[1],
                   raw.at<cv::Vec3b>( i, j )[2] );
        DebugInfo( "(%.2f %.2f %.2f)",
                   img( i, j )[0],
                   img( i, j )[1],
                   img( i, j )[2] );
        ResumeOnRet();
      }
    }
  }

  // for ( int i=0; i<raw.rows; i++ ) {
  //   for ( int j=0; j<raw.cols; j++ ) {
  //     double norm = img(i,j)[0] * img(i,j)[0] + img(i,j)[1]*img(i,j)[1];
  //     if ( norm > 10000.0 ) {
  //       DebugInfo( "%hhu %hhu %hhu", 
  //                  lab.at<cv::Vec3b>( i, j )[0],
  //                  lab.at<cv::Vec3b>( i, j )[1],
  //                  lab.at<cv::Vec3b>( i, j )[2] );
  //       DebugInfo( "(%.2f %.2f) (%.2f %.2f)", 
  //                  img( i, j )[0],
  //                  img( i, j )[1],
  //                  img( i, j )[2],
  //                  img( i, j )[3] );
  //       ResumeOnRet();
  //     }
  //   }
  // }
  return 0;
}

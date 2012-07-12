/*********************************************************************************
 * File: PatchShift.cpp
 * Description: Sample program for using 2d library. Search for the nearest neighbor
 *              in image A from image B. Search is limited within a range.
 * by BreakDS, University of Wisconsin Madison, Wed Jul 11 10:45:21 CDT 2012
 *********************************************************************************/

#include <utility>
#include "data/features.hpp"
#include "opencv2/opencv.hpp"

using namespace PatTk;
using std::pair;


Image<LabCell,std::pair<int,int> > InitCellImage( const cv::Mat& mat )
{
  Image<LabCell, std::pair<int,int> > img( mat.rows, mat.cols );
  cv::Mat lab;
  cvtColor( mat, lab, CV_BGR2Lab );

  int i = 0;
  for ( auto it = lab.begin<cv::Vec3b>(), end = lab.end<cv::Vec3b>(); it != end; it ++ ) {
    img.setCell( i, LabCell( (*it)[0], (*it)[1], (*it)[2] ) );
    img.setVal( i, std::make_pair( 0, 0 ) );
    i++;
  }
  
  return img;
}



int main( int argc, char **argv )
{
  
  if ( argc < 2 ) {
    Error( "Not enough arguments." );
    exit( -1 );
  }
  cv::Mat mat = cv::imread( argv[1] );
  
  Image<LabCell, std::pair<int,int> > img = InitCellImage( mat );
  for ( int y=0; y<img.rows; y++ ) {
    for ( int x=0; x<img.cols; x++ ) {
      img(y,x).Summary();
      char ch;
      scanf( "%c", &ch );
    }
  }
  img.Summary();
  return 0;
}


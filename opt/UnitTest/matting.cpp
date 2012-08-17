/*********************************************************************************
 * File: matting.hpp
 * Description: UnitTest for Distance Transform Loopy Belief Propagation
 * by BreakDS, @ University of Wisconsin-Madison, Fri Aug 17 17:58:10 CDT 2012
 *********************************************************************************/
#include <cstdio>
#include "../BP.hpp"
#include "opencv2/opencv.hpp"




void display( uchar *img, int h, int w )
{
  cv::Mat canvas( h, w, CV_8UC1 );
  for ( int i=0; i<h; i++ ) {
    for ( int j=0; j<w; j++ ) {
      canvas.at<uchar>( i, j ) = img[ i * w + j ];
    }
  }
  cv::imshow( "canvas", canvas );
  cv::waitKey();
}


int main()
{
  // Generate A 100 x 100 image
  int N = 100;
  uchar img[N*N];
  for ( int i=0; i<N; i++ ) {
    for ( int j=0; j<N; j++ ) {
      float st = N * sin( 3.14 / N * j ) * 0.7;
      if ( static_cast<float>(i) < st ) {
        img[i*N+j] = rand() % 40 + 210;
      } else {
        img[i*N+j] = rand() % 50;
      }
    }
  }

  display( img, N, N );

  // Preparing the candidate
  uchar feature[N*N*2];
  float labels[N*N*2*1];
  for ( int i=0; i<N*N; i++ ) {
    feature[i*2 + (i&1)] = 200;
    feature[i*2 + (1-(i&1))] = 50;
    labels[i*2 + (i&1)] = 255;
    labels[i*2 + (1-(i&1))] = 0;
  }

  optmize::Options options;
  options.maxIter = 10;
  options.numHypo = 1;

  float D[N*N*2];
  
  
  

  return 0;
}



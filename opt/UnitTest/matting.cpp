/*********************************************************************************
 * File: matting.cpp
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

template <typename floating=float>
class RandProj : public optimize::AbstractRandHash<floating>
{
public:
  vector<floating> coeff;
  void shuffle( int dim ) 
  {
    static const float ub = static_cast<double>( RAND_MAX );
    coeff.resize(dim);
    for ( int i=0; i<dim; i++ ) {
      coeff[i] = rand() / ub;
    }
  }
  
  floating operator()( const floating* a, int dim )
  {
    floating sum = 0.01;
    for ( int i=0; i<dim; i++ ) {
      sum += a[i] * coeff[i];
    }
    return sum;
  }
};


int main()
{
  // Generate An M x N image
  int M = 100;
  int N = 50;
  uchar img[N*M];
  for ( int i=0; i<M; i++ ) {
    for ( int j=0; j<N; j++ ) {
      float st = M * sin( 3.14 / N * j ) * 0.7;
      if ( static_cast<float>(i) < st ) {
        //        img[i*N+j] = rand() % 40 + 210;
        img[i*N+j] = 255;
      } else {
        //        img[i*N+j] = rand() % 50;
        img[i*N+j] = 0;
      }
    }
  }

  // Preparing the candidate
  uchar feature[N*M*2];
  float labels[N*M*2*1];
  for ( int i=0; i<N*M; i++ ) {
    labels[i*2] = 255; //rand() % 10 + 210;
    labels[i*2+1] = 0; //rand() % 20;
    feature[i*2] = static_cast<uchar>( labels[i*2] );
    feature[i*2+1] = static_cast<uchar>( labels[i*2+1] );
  }

  optimize::Options options;
  options.maxIter = 10;
  options.numHypo = 3;

  float D[N*N*2];
  for ( int i=0; i<N*M; i++ ) {
    D[i*2] = (feature[i*2] - img[i]) * (feature[i*2] - img[i] );
    D[i*2+1] = (feature[i*2+1] - img[i]) * (feature[i*2+1] - img[i] );
  }
  

  int result[M*N];
  
  optimize::LoopyBP<RandProj<float>,optimize::FDT<float>,float>( D, labels, 1e-10, M, N, 2, 1, result, options );
  
  uchar labeled[M*N];
  for ( int i=0; i<M*N; i++ ) {
    if ( result[i] == 0 ) {
      labeled[i] = 255;
    } else {
      labeled[i] = 0;
    }
  }
  
  display( labeled, M, N );

  return 0;
}



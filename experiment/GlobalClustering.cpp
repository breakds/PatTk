/*********************************************************************************
 * File: GlobalClustering.hpp
 * Description: Plane-Fitting on the set of whole possible matching transformation,
 *              where the peaks will be pulled out to form the candidates
 * by BreakDS, @ University of Wisconsin-Madison, Tue Oct 23 16:26:37 CDT 2012
 *********************************************************************************/

#include <vector>
#include <string>
#include "LLPack/utils/Environment.hpp"
#include "LLPack/algorithms/heap.hpp"
#include "LLPack/algorithms/list.hpp"
#include "../interfaces/cv_interface.hpp"
#include "../graph/Graph.hpp"

#define RANDOM() ( static_cast<float>( rand() ) / RAND_MAX )
#define PERTURB(x) ( ( ( static_cast<float>( rand() ) / RAND_MAX ) - 0.5 ) * x )


using namespace PatTk;
using namespace EnvironmentVariable;

/* Fitting K transformations in N pairs of matches (x,y), where both x
 * and y are 2D coordinates. The result will be a K*4-long array
 * describing the K transformations, and an N-long array describing
 * the assignment of transformations for each pair of match.
 */


struct Options {
  int w0, w1, h0, h1;
  int maxIter;
};


void TransformFitting( const float* x, const float* y, int N, int K, Options options,
                       int* &assign, float* &transforms  )
{
  if ( nullptr == assign ) {
    assign = new int[N];
  }

  if ( nullptr == transforms ) {
    transforms = new float[N*4];
  }


  // solve:
  // min sum_i sum_k pi_{i,k} || H_i * t_k - b_i ||^2
  // where,
  // H_i = | x_1(i) x_2(i)   1 0 |
  //       | x_2(i) -x_1(i)  0 1 |
  // t_k is of 4-dim
  // b_i = | y_1(i) |
  //       | y_2(i) |


  // solution:
  // ====================================================================
  // Fix pi_{i,k}:
  // t_k = [sum_i (H_i' * H_i)]^{-1} * [sum_i (Hi' * b_i)], where i \in k
  // ====================================================================
  // Fix t_k:
  // pi_{i,k} = 1 if and only if k = arg min_{v} || H_i * t_v - b_i ||^2


  // Initialization of H;
  cv::Mat *H = new cv::Mat[N];
  {
    const float *xp = x;
    for ( int i=0; i<N; i++ ) {
      // HTH[i] = H[i]' * H[i]
      //          |  x^2     0      x1      x2  |
      //        = |   0     x^2     x2     -x1  |
      //          |  x1     x2      1       0   |
      //          |  x2     -x1     0       1   |

      H[i].create( 2, 4, CV_32FC1 );
      H[i].at<float>( 0, 0 ) = xp[0];
      H[i].at<float>( 0, 1 ) = xp[1];
      H[i].at<float>( 0, 2 ) = 1.0f;
      H[i].at<float>( 0, 3 ) = 0.0f;
      H[i].at<float>( 1, 0 ) = xp[1];
      H[i].at<float>( 1, 1 ) = -xp[0];
      H[i].at<float>( 1, 2 ) = 0.0f;
      H[i].at<float>( 1, 3 ) = 1.0f;
      xp+=2;
    }
  }
  

  // Initialization of HTH
  cv::Mat *HTH = new cv::Mat[N];
  for ( int n=0; n<N; n++ ) {
    cv::mulTransposed( H[n], HTH[n], true );
  }

  // Initialization of y's
  cv::Mat *yvec = new cv::Mat[N];
  {
    const float *yp = y;
    for ( int i=0; i<N; i++ ) {
      yvec[i].create( 2, 1, CV_32FC1 );
      yvec[i].at<float>( 0, 0 ) = *(yp++);
      yvec[i].at<float>( 1, 0 ) = *(yp++);
    }
  }

  // Initialization of HTy
  cv::Mat *HTy = new cv::Mat[N];
  for ( int i=0; i<N; i++ ) {
    cv::Mat tmp;
    cv::transpose( H[i], tmp );
    HTy[i] = tmp * yvec[i];
  }

  // Initialization of assignment
  for ( int i=0; i<N; i++ ) {
    assign[i] = i % K;
  }
  

  // aux arrays
  cv::Mat A[K];
  cv::Mat b[K];
  cv::Mat trans[K];
  // main interations
  for ( int iter=0; iter<options.maxIter; iter++ ) {
    // update "centers"
    for ( int k=0; k<K; k++ ) {
      A[k] = cv::Mat::zeros( 4, 4, CV_32FC1 );
      b[k] = cv::Mat::zeros( 4, 1, CV_32FC1 );
    }

    for ( int i=0; i<N; i++ ) {
      A[assign[i]] += HTH[i];
      b[assign[i]] += HTy[i];
    }

    for ( int k=0; k<K; k++ ) {
      // for ( int i=0; i<4; i++ ) {
      //   for ( int j=0; j<4; j++ ) {
      //     printf( "%.2f\t", A[k].at<float>( i, j ) );
      //   }
      //   printf( "\n" );
      // }
      // char ch;
      // scanf( "%c", &ch );
      // printf( "determinant: %.4lf\n", cv::determinant( A[k] ) );
      // for ( int i=0; i<4; i++ ) {
      //   printf( "%.2f ", b[k].at<float>( i, 0 ) );
      // }
      // printf( "\n" );
      cv::solve( A[k], b[k], trans[k] );
    }

    // update assignments and energy
    double energy = 0.0;
    for ( int i=0; i<N; i++ ) {
      assign[i] = -1;
      double min = 0.0;
      for ( int k=0; k<K; k++ ) {
        cv::Mat diff = H[i] * trans[k] - yvec[i];
        double dist = cv::norm( diff );
        if ( -1 == assign[i] || dist < min ) {
          min = dist;
          assign[i] = k;
        }
      }
      energy += min * min;
    }
    Info( "iter %d: energy = %.10lf\n", iter, energy );
  }

  // assignment to transform array
  {
    float *transformp = transforms;
    for ( int k=0; k<K; k++ ) {
      for ( int j=0; j<4; j++ ) {
        *(transformp++) = trans[k].at<float>( j, 0 );
      }
    }
  }

  DeleteToNullWithTestArray( H );
  DeleteToNullWithTestArray( yvec );
  DeleteToNullWithTestArray( HTH );
  DeleteToNullWithTestArray( HTy );
}


/*
void experiment()
{
  srand(19871017);
  int N = 100;
  float x[N*2];
  float y[N*2];
  
  for ( int i=0; i<N; i++ ) {
    x[i*2] = static_cast<float>( rand() ) / RAND_MAX * 600.0;
    x[i*2 + 1] = static_cast<float>( rand() ) / RAND_MAX * 400.0;
  
    int k = rand() % 3;
    if ( 0 == k ) {
      y[i*2] = x[i*2] + 10.0 + PERTURB(5.0);
      y[i*2+1] = x[i*2+1] + 20.0 + PERTURB(5.0);
    } else if ( 1 == k ) {
      y[i*2] = x[i*2] + 30.0 + PERTURB(5.0);
      y[i*2+1] = x[i*2+1] + 50.0 + PERTURB(5.0);
    } else {
      y[i*2] = x[i*2] - 15.0 + PERTURB(5.0);
      y[i*2+1] = x[i*2+1] - 10.0 + PERTURB(5.0);
    }
  }

  int *assign = nullptr;
  float *trans = nullptr;
  Options options;
  options.maxIter = 10;
  TransformFitting( x, y, N, 3, options, assign, trans );

  for ( int k=0; k<3; k++ ) {
    printf( "%.2f, %.2f, %.2f, %.2f\n", trans[k*4], trans[k*4+1], trans[k*4+2], trans[k*4+3] );
  }

  DeleteToNullWithTestArray( assign );
  DeleteToNullWithTestArray( trans );
}
*/




int main( int argc, char **argv )
{
  if ( argc < 2 ) {
    Error( "Missing configuration file in options. (Global.conf)" );
    exit( -1 );
  }

  env.parse( argv[1] );
  env.Summary();

  PatGraph graph( env["graph-file"] );


  // prepare x array and y array
  std::vector<float> x;
  std::vector<float> y;
  int radius = env["patch-w"] >> 1;
  for ( int i=0; i<graph.rows - env["patch-w"]; i++ ) {
    for ( int j=0; j<graph.cols - env["patch-w"]; j++ ) {
      for ( auto& ele : graph(i,j) ) {
        x.push_back( static_cast<float>( i + radius ) );
        x.push_back( static_cast<float>( j + radius ) );
        y.push_back( static_cast<float>( ele.y + radius * ele.scale ) );
        y.push_back( static_cast<float>( ele.x + radius * ele.scale ) );
      }
    }
  }


  int *assign = nullptr;
  float *trans = nullptr;
  Options options;
  options.maxIter = 15;

  Info( "Starting Optimization: %ld pairs.", x.size() >> 1 );
  
  TransformFitting( &x[0], &y[0], x.size() >> 1, env["cluster-num"], options, assign, trans );


  // statitiscs
  int count[static_cast<int>(env["cluster-num"])];
  for ( int i=0; i<env["cluster-num"]; i++ ) {
    count[i] = 0;
  }

  for ( int i=0; i<static_cast<int>( x.size() >> 1 ); i++ ) {
    count[assign[i]]++;
  }

  for ( int k=0; k<env["cluster-num"]; k++ ) {
    printf( "%.2f, %.2f, %.2f, %.2f: %d\n", trans[k*4], trans[k*4+1], trans[k*4+2], trans[k*4+3], count[k] );
  }

  DeleteToNullWithTestArray( assign );
  DeleteToNullWithTestArray( trans );


  
  return 0;
}

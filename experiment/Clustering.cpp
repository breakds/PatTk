/*********************************************************************************
 * File: Clustering.hpp
 * Description: Clustering of nearest neighbor offsets
 * by BreakDS, @ University of Wisconsin-Madison, Thu Oct 18 15:36:25 CDT 2012
 *********************************************************************************/

#include <vector>
#include <string>
#include "LLPack/utils/Environment.hpp"
#include "LLPack/algorithms/heap.hpp"
#include "LLPack/algorithms/list.hpp"
#include "../interfaces/cv_interface.hpp"
#include "../graph/Graph.hpp"

using namespace PatTk;
using namespace EnvironmentVariable;

// template <typename valueType>
// void slowPeak(  const cv::Mat &src, cv::Mat &dst, int radius=4 )
// {
//   dst = src.clone();
//   for ( int i=0; i<src.rows; i++ ) {
//     for ( int j=0; j<src.cols; j++ ) {
//       int beginX = ( j - radius >= 0 ) ? ( j - radius ) : 0;
//       int endX = ( j + radius < src.cols ) ? ( j + radius ) : src.cols - 1;
//       int beginY = ( i - radius >= 0 ) ? ( i - radius ) : 0;
//       int endY = ( i + radius < src.cols ) ? ( i + radius ) : src.rows - 1;
      
//       valueType max = src.at<valueType>( beginY, beginX );

//       for ( int y=beginY; y<=endY; y++ ) {
//         for ( int x=beginX; x<=endX; x++ ) {
//           if ( src.at<valueType>( y, x ) > max ) {
//             max = src.at<valueType>( y, x );
//           }
//         }
//       }
//       dst.at<valueType>( i ,j ) = max;
//     }
//   }
// }

template <typename valueType>
void GetPeakMap( const cv::Mat &src, cv::Mat &dst, int radius=4 )
{
  cv::Mat colPos( src.rows, src.cols, CV_32SC1 );

  // Prepare stacks;
  int K = (radius<<1) + 1;
  Circular<valueType> valStack(K);
  Circular<int> posStack(K);
  
  // Per Row Scan:
  for ( int i=0; i<src.rows; i++ ) {

    valStack.clear();
    posStack.clear();
    
    for ( int j=0; j<src.cols+radius; j++ ) {
      if ( j < src.cols ) {
        valueType val = src.at<valueType>( i, j );

        // pop stack
        while ( ( !valStack.empty() ) && val > valStack.back() ) {
          valStack.pop_back();
          posStack.pop_back();
        }

        // push val
        valStack.push_back( val );
        posStack.push_back( j );

      }
      
      // If not within range, pop front
      if ( posStack.front() <= j - K ) {
        valStack.pop_front();
        posStack.pop_front();
      }
      
      if ( j >= radius ) {
        colPos.at<int>( i, j-radius ) = posStack.front();
      }
    }
  }

  cv::Mat rowPos( src.rows, src.cols, CV_32SC1 );

  // Per Column Scan:
  for ( int j=0; j<src.cols; j++ ) {

    valStack.clear();
    posStack.clear();
    
    for ( int i=0; i<src.rows+radius; i++ ) {
      if ( i < src.rows ) {
        valueType val = src.at<valueType>( i, colPos.at<int>( i, j ) );

        // pop stack
        while ( ( !valStack.empty() ) && val > valStack.back() ) {
          valStack.pop_back();
          posStack.pop_back();
        }

        // push val
        valStack.push_back( val );
        posStack.push_back( i );

      }
      
      // If not within range, pop front
      if ( posStack.front() <= i - K ) {
        valStack.pop_front();
        posStack.pop_front();
      }
      
      if ( i >= radius ) {
        rowPos.at<int>( i-radius, j ) = posStack.front();
      }
    }
  }
  


  

  dst.create( src.rows, src.cols, CV_32SC2 );
  
  for ( int i=0; i<src.rows; i++ ) {
    for ( int j=0; j<src.cols; j++ ) {
      int y = rowPos.at<int>( i, j );
      int x = colPos.at<int>( y, j );
      dst.at<cv::Vec2i>( i, j )[0] = y;
      dst.at<cv::Vec2i>( i, j )[1] = x;
    }
  }
}


class OffsetHist
{
private:
  cv::Mat map;
  int shiftY, shiftX;
public:
  // (h0, w0) is the size of the target image
  // (h1, w1) is the size of the source image
  // offset = (y1,x1) - (y0,x0)

  int rows, cols;

  OffsetHist( int h0, int w0, int h1, int w1 ) : shiftY(h0-1), shiftX(w0-1), rows(h0+h1-1), cols(w0+w1-1)
  {
    map = cv::Mat::zeros( rows, cols, CV_32FC1 );
  }

  inline void drop( int dy, int dx )
  {
    int y = dy + shiftY;
    int x = dx + shiftX;
    if ( 0 <= y && y<rows && 0<=x && x<cols ) {
      map.at<float>( dy+shiftY, dx+shiftX ) += 1.0f;
    }
  }

  inline void undrop( int dy, int dx )
  {
    int y = dy + shiftY;
    int x = dx + shiftX;
    if ( 0 <= y && y<rows && 0<=x && x<cols ) {
      map.at<float>( dy+shiftY, dx+shiftX ) -= 1.0f;
    }
  }

  inline std::vector<std::pair<int,int> > getPeaks( int k ) {

    // value Map = smoothed
    cv::Mat smoothed;

    cv::GaussianBlur( map, smoothed, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    cv::Mat peakMap;
    GetPeakMap<float>( smoothed, peakMap );

    
    // Filter 1: Get Peaks
    std::vector< std::pair<int,int> > collect;
    collect.clear();
    for ( int i=0; i<peakMap.rows; i++ ) {
      for ( int j=0; j<peakMap.cols; j++ ) {
        int y = peakMap.at<cv::Vec2i>( i, j )[0];
        int x = peakMap.at<cv::Vec2i>( i, j )[1];
        if ( y == i && x == j ) {
          collect.push_back( std::make_pair( y, x ) );
        }
      }
    }

    

    // filter 2: Get top peaks
    heap<float,std::pair<int,int> > ranker( k );
    for ( auto& ele : collect ) {
      int y = ele.first;
      int x = ele.second;
      ranker.add( -smoothed.at<float>( y, x ), std::make_pair( y, x ) );
    }

    
    std::vector< std::pair<int,int> > peaks;
    peaks.clear();
    peaks.reserve( k );
    for ( int i=0; i<ranker.len; i++ ) {
      int y = ranker[i].first;
      int x = ranker[i].second;
      peaks.push_back( std::make_pair( y - shiftY, x - shiftX ) );
    }
    
    return peaks;

  }
  
};

int main( int argc, char **argv )
{
  if ( argc < 2 ) {
    Error( "Missing configuration file in options." );
    exit( -1 );
  }

  env.parse( argv[1] );
  env.Summary();


  int wndSize = env["clustering-window"];
  int wndRadius = wndSize >> 1;
  
  PatGraph graph( env["graph-file"] );
  PatGraph result( graph.rows, graph.cols );


  for ( int i=0; i<graph.rows; i++ ) {
    Info( "%d/%d", i, graph.rows );
    int beginY = ( i >= wndRadius ) ? i - wndRadius : 0;
    int endY = ( i + wndRadius < graph.rows ) ? i + wndRadius : graph.rows - 1;
    for ( int j=0; j<graph.cols; j++ ) {
      int beginX = ( j >= wndRadius ) ? j - wndRadius : 0;
      int endX = ( j + wndRadius < graph.cols ) ? j + wndRadius : graph.cols - 1;
      OffsetHist h(320,240,320,240);
      int k = 0;
      // debugging
      for ( int y=beginY; y<=endY; y++ ) {
        for ( int x=beginX; x<=endX; x++ ) {
          k++;
          h.drop( static_cast<int>( graph( y, x )[0].y - i + .5),
                  static_cast<int>( graph( y, x )[0].x - j + .5) );
        }
      }
      
      std::vector<std::pair<int,int> > dominants = std::move( h.getPeaks(5) );

      if ( 0 == dominants.size() ) {
        Error( "no peaks for (%d,%d)." );
      }

      for ( auto& ele : dominants ) {
        result[i * graph.cols + j].push_back( PatLoc( graph(i,j)[0].index,
                                                      ele.first + i,
                                                      ele.second + j,
                                                      1.0,
                                                      0.0,
                                                      0.0 ) );
      }
    }
  }

  std::string savepath = env["output"];
  result.write( savepath );
  
  return 0;
}

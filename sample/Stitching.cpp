/*********************************************************************************
 * File: Stitching.hpp
 * Description: Visualize the stitching of graph builder
 * by BreakDS, @ University of Wisconsin-Madison, Thu Sep 27 16:05:29 CDT 2012
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

Image<BGRCell,int,false>::Patch Loc2Patch( const Image<BGRCell,int,false> &img,
                                           const PatLoc& candidate )
{
  static int radius = env["patch-w"] >> 1;

  double ang = - candidate.rotation;
  double cosa = cos( ang ) * candidate.scale;
  double sina = sin( ang ) * candidate.scale;
  
  double y1 = ( -radius * cosa - radius * sina + ( candidate.y + radius ) );
  double x1 = ( radius * sina - radius * cosa + ( candidate.x + radius ) );

  return img.Spawn( y1, x1, 1.0 / candidate.scale, ang / M_PI * 180.0 );
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

  // Stitching with blending
  int counts[graph.rows][graph.cols];
  float values[graph.rows][graph.cols][3];
  for ( int i=0; i<graph.rows; i++ ) {
    for ( int j=0; j<graph.cols; j++ ) {
      counts[i][j] = 0;
      for ( int k=0; k<3; k++ ) {
        values[i][j][3] = 0;
      }
    }
  }


  vector<int> imageIDs;
  imageIDs.resize( graph.rows * graph.cols );
  for ( int i=0; i<graph.rows*graph.cols; i++ ) {
    imageIDs[i] = graph(i)[0].index;
  }
  vector<int> sorted = std::move( sorting::index_sort( imageIDs ) );



  Image<BGRCell, int, false> image;
  
  int currentImgID = -1;
  for ( auto& i : sorted ) {
  // for ( int i=0; i<graph.rows*graph.cols; i++ ) {
    int y = i / graph.cols;
    int x = i % graph.cols;
    if ( graph(i)[0].index != currentImgID ) {
      currentImgID = graph(i)[0].index;
      cv::Mat raw = cv::imread( strf( "%s/%s_L.png", env["dataset"].c_str(),
                                      imgList[currentImgID].c_str() ) );
      if ( raw.empty() ) {
        Error( "cannot open image %s/%s.png", env["dataset"].c_str(),
               imgList[currentImgID].c_str() );
        return -1;
      }
      image = std::move( cvFeatGen<BGRCell, int, false>::gen( raw ) );
      image.SetPatchParameter( env["patch-w"], env["patch-w"] );
    }

    
    Image<BGRCell, int, false>::Patch candPatch= Loc2Patch( image, graph(i)[0] );

    int k = 0;
    for ( int dy=0; dy<env["patch-w"]; dy++ ) {
      for ( int dx=0; dx<env["patch-w"]; dx++ ) {
        if ( y + dy < graph.rows && x + dx < graph.cols ) {
          counts[y+dy][x+dx]++;
          for ( int ch=0; ch<3; ch++ ) {
            values[y+dy][x+dx][ch] += candPatch[k];
            k++;
          }
        } else {
          k += 3;
        }

      }
    }
  }


  cv::Mat result( graph.rows, graph.cols, CV_8UC3 );

  for ( int i=0; i<graph.rows; i++ ) {
    for ( int j=0; j<graph.cols; j++ ) {
      for ( int k=0; k<3; k++ ) {
        result.at<cv::Vec3b>(i,j)[k] = static_cast<uchar>( values[i][j][k] / counts[i][j] );
      }
    }
  }

  cv::imwrite( "tmp.png", result );

  imshow( "result", result );

  while ( cv::waitKey(30) != 27 );


  
  return 0;
  
}

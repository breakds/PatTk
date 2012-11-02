#include <vector>
#include <string>

#include "LLPack/utils/Environment.hpp"
#include "LLPack/utils/extio.hpp"
#include "../interfaces/cv_interface.hpp"

using namespace PatTk;
using namespace EnvironmentVariable;






void dot( std::string wnd,
          cv::Mat &img,
          std::vector<std::pair<float,float> >& dots )
{
  cv::Mat canvas = img.clone();
  for ( auto& ele : dots ) {
    int y = static_cast<int>( ele.first );
    int x = static_cast<int>( ele.second );
    if ( 0 <= y && y < canvas.rows &&
         0 <= x && x < canvas.cols ) {
      canvas.at<cv::Vec3b>(y,x)[0] = 0;
      canvas.at<cv::Vec3b>(y,x)[1] = 255;
      canvas.at<cv::Vec3b>(y,x)[2] = 0;
    }
  }
  cv::imshow( wnd, canvas );
}

struct MouseParam
{
  std::vector< std::pair< float, float> > &orgDots;
  std::vector< std::pair< float, float> > &refDots;
  cv::Mat &org;
  cv::Mat &ref;
  std::vector<AffineTransform> &trans;
  bool drawing;
  int &tIndex;
};


static void MouseCallback( int event, int x, int y, int __attribute__((__unused__)) flags, void *param )
{
  if ( event == CV_EVENT_LBUTTONDOWN ) {
    MouseParam* p = (MouseParam*) param;
    p->drawing = true;
  } else if ( CV_EVENT_MOUSEMOVE == event ) {
    MouseParam* p = (MouseParam*) param;
    if ( p->drawing ) {
      p->orgDots.push_back( std::make_pair( static_cast<float>( y ),
                                            static_cast<float>( x ) ) );


      float y1 = 0.0f;
      float x1 = 0.0f;
      p->trans[p->tIndex].apply( y, x, y1, x1 );
      p->refDots.push_back( std::make_pair( y1, x1 ) );
    }
  } else if ( CV_EVENT_LBUTTONUP == event ) {
    MouseParam* p = (MouseParam*) param;
    p->drawing = false;
    dot( "org", p->org, p->orgDots );
    dot( "ref", p->ref, p->refDots );
  }
}




int main( int argc, char **argv )
{
  if ( argc < 2 ) {
    Error( "Missing configuration file in options. (show.conf)" );
    exit( -1 );
  }

  env.parse( argv[1] );
  env.Summary();

  std::vector<std::string> imgList = std::move( readlines( strf( "%s/%s", env["dataset"].c_str(),
                                                                 env["list-file"].c_str() ) ) );

  cv::Mat org = cv::imread( strf( "%s/%s.png", env["dataset"].c_str(),
                                  imgList[env["original"]].c_str() ) );
  cv::Mat ref = cv::imread( strf( "%s/%s.png", env["dataset"].c_str(),
                                  imgList[env["reference"]].c_str() ) );

  std::vector< std::pair< float, float> > orgDots;
  std::vector< std::pair< float, float> > refDots;

  int K = 0;
  std::vector<AffineTransform> trans;
  WITH_OPEN( in, env["transform"].c_str(), "r" );
  fread( &K, sizeof(int), 1, in );
  trans.resize( K );
  for ( int k=0; k<K; k++ ) {
    
    trans[k].read( in );
  }
  END_WITH( in );

  int tIndex = 0;
  
  MouseParam mouseParam = { orgDots, refDots, org, ref, trans, false, tIndex };

  cv::imshow( "org", org );
  cv::imshow( "ref", ref );

  cv::setMouseCallback( "org", MouseCallback, &mouseParam );

  char ch = 0;

  while ( 27 != (ch = cv::waitKey( 100 ) ) ) {
    if ( ch == 81 ) {
      if ( tIndex > 0 ) {
        tIndex--;
        refDots.clear();
        for ( auto& every : orgDots ) {
          float y1 = 0.0;
          float x1 = 0.0;
          trans[tIndex].apply( every.first, every.second, y1, x1 );
          refDots.push_back( std::make_pair( y1, x1 ) );
        }
        dot( "org", org, orgDots );
        dot( "ref", ref, refDots );
      }
      printf( "tIndex: %d\n", tIndex );
    } else if ( ch == 83 ) {
      if ( tIndex < static_cast<int>( trans.size() - 1 ) ) {
        tIndex++;
        refDots.clear();
        for ( auto& every : orgDots ) {
          float y1 = 0.0;
          float x1 = 0.0;
          trans[tIndex].apply( every.first, every.second, y1, x1 );
          refDots.push_back( std::make_pair( y1, x1 ) );
        }
        dot( "org", org, orgDots );
        dot( "ref", ref, refDots );
      }
      printf( "tIndex: %d\n", tIndex );
    }
  }



  return 0;
  
}


           


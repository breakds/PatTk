/*********************************************************************************
 * File: opencv_aux.hpp
 * Author: BreakDS <breakds@cs.wisc.edu>
 *         University of Wisconsin-Madison, 2012
 * Description: Provide auxiliary routines as interfaces between opencv and PatTk
 *********************************************************************************/


#pragma once
#include <string>
#include <type_traits>
#include "opencv2/opencv.hpp"
#include "LLPack/utils/candy.hpp"
#include "LLPack/utils/time.hpp"
#include "../graph/Trans.hpp"
#include "../data/FeatImage.hpp"
#include "../data/vector.hpp"

#define _USE_MATH_DEFINES

namespace PatTk
{

  enum featEnum { BGR, Lab, HOG, DEFAULT_FEAT };
  
  /* anonymous namespace for helper functions */
  namespace
  {
    template <featEnum featType>
    class GetDataType{
    public:
      typedef void type;
    };

    template <>
    class GetDataType<BGR>
    {
    public:
      typedef uchar type;
    };

    template <>
    class GetDataType<Lab>
    {
    public:
      typedef uchar type;
    };

    template <>
    class GetDataType<HOG>
    {
    public:
      typedef float type;
    };

  }



  
  template <featEnum featType> struct FeatOptions {};

  
  template<>
  struct FeatOptions<HOG>
  {
    int orientation_bins;
    int cell_side;
    bool enable_color;
    bool gaussian_filter;
    FeatOptions() : orientation_bins(9), cell_side(5),
                    enable_color(true), gaussian_filter(false)
    {}

  };


  
  

  template <featEnum featType=DEFAULT_FEAT>
  class cvFeat
  {
  public:

    static FeatOptions<featType> options;
    
    template <featEnum T=featType>
    static FeatImage<unsigned char> gen( std::string filename, ENABLE_IF(DEFAULT_FEAT==T) )
    {
      return FeatImage<unsigned char>();
    }
    
    // +--------------------------------------------------+
    // |  BGR Feature Generator                           |
    // +--------------------------------------------------+
    template <featEnum T=featType>
    static FeatImage<uchar> gen( cv::Mat raw, ENABLE_IF( BGR == T ) )
    {
      if ( 1 == raw.channels() ) {
        FeatImage<uchar> img( raw.rows, raw.cols, 3 );
        uchar *img_ptr = img[0];
        for ( int i=0; i<raw.rows; i++ ) {
          uchar *raw_ptr = raw.ptr<uchar>(i);
          for ( int j=0; j<raw.cols; j++ ) {
            *(img_ptr++) = *raw_ptr;
            *(img_ptr++) = *raw_ptr;
            *(img_ptr++) = *raw_ptr;
            raw_ptr++;
          }
        }
        return img;
      } else if ( 3 == raw.channels() ) {
        FeatImage<uchar> img( raw.rows, raw.cols, 3 );
        uchar *img_ptr = img[0];
        size_t row_size = sizeof(uchar) * 3 * raw.cols;
        int row_stride = 3 * raw.cols;
        for ( int i=0; i<raw.rows; i++ ) {
          uchar *raw_ptr = raw.ptr<uchar>(i);
          memcpy( img_ptr, raw_ptr, row_size );
          img_ptr += row_stride;
        }
        return img;
      }

      Error( "cvFeat<BGR>::gen()   Bad number of channels: %d\n", raw.channels() );
      exit( -1 );
    }

    template <featEnum T=featType>
    static FeatImage<uchar> gen( std::string filename, ENABLE_IF( BGR == T ) )
    {
      cv::Mat raw = cv::imread( filename );
      if ( raw.empty() ) {
        Error( "cvFeat<BGR::gen()   Failed to load image %d", filename.c_str() );
        exit( -1 );
      }
      return gen<BGR>( raw );
    }


    template <featEnum T=featType>
    static FeatImage<uchar> gen( std::string filename, int scales, float base, ENABLE_IF( BGR == T ) )
    {
      
      cv::Mat raw = cv::imread( filename );
      if ( raw.empty() ) {
        Error( "cvFeat<BGR::gen()   Failed to load image %d", filename.c_str() );
        exit( -1 );
      }
      
      std::vector<FeatImage<uchar> > imgs;

      imgs.resize( ( scales << 1 ) + 1 );
      
      imgs[scales] = std::move( gen<BGR>( raw ) );

      cv::Mat org = raw.clone();
      cv::Mat tmp;
      float scale = base;
      for ( int i=-1; i>=-scales; i-- ) {
        cv::resize( org, tmp, cv::Size( org.cols * scale, org.rows * scale ) );
        scale = scale * base;
        imgs[scales+i] = std::move( gen<BGR>( tmp ) );
      }


      
      scale = 1.0 / base;
      for ( int i=1; i<=scales; i++ ) {
        cv::resize( org, tmp, cv::Size( org.cols * scale, org.rows * scale ) );
        scale = scale / base;
        imgs[scales+i] = std::move( gen<BGR>( tmp ) );
      }
      
      return FeatImage<uchar>( std::move(imgs), base );
    }



    // +--------------------------------------------------+
    // |  Lab Feature Generator                           |
    // +--------------------------------------------------+
    template <featEnum T=featType>
    static FeatImage<uchar> gen( std::string filename, ENABLE_IF( Lab == T ) )
    {
      cv::Mat raw = cv::imread( filename );
      if ( raw.empty() ) {
        Error( "cvFeat<Lab>::gen()   Failed to load image %s", filename.c_str() );
        exit( -1 );
      }
      
      if ( 3 == raw.channels() ) {
        cv::Mat lab;
        cv::cvtColor( raw, lab, CV_BGR2Lab );
        FeatImage<uchar> img( lab.rows, lab.cols, 3 );
        uchar *img_ptr = img[0];
        size_t row_size = sizeof(uchar) * 3 * lab.cols;
        int row_stride = 3 * lab.cols;
        for ( int i=0; i<lab.rows; i++ ) {
          uchar *lab_ptr = lab.ptr<uchar>(i);
          memcpy( img_ptr, lab_ptr, row_size );
          img_ptr += row_stride;
        }
        return img;
      }

      Error( "cvFeat<Lab>::gen()   Bad number of channels: %d\n", raw.channels() );
      exit( -1 );
    }




    



    template <featEnum T=featType>
    inline static void Angle2Bin( float angle, int &lo_bin, int &hi_bin, float &lo_wt, float &hi_wt,
                                  ENABLE_IF( HOG == T ) )
    {
      /* centers are 0, span, 2*span, 3*span, ..., (bins-1) * span,
       * where bins * span and 0 are coincident centers. each center
       * spans +-(span/2), and span (bin_span) is defined right below.
       */
      float bin_span = M_PI / options.orientation_bins;
      float tmp = angle < 0.0 ? M_PI + angle : angle;

      float pos = tmp / bin_span;
      lo_bin = static_cast<int>( pos );
      if ( lo_bin == options.orientation_bins ) lo_bin--;
      hi_bin = lo_bin + 1;
      if ( options.orientation_bins == hi_bin ) hi_bin = 0;
      hi_wt = pos - lo_bin;
      lo_wt = 1.0 - hi_wt;
    }

    // +--------------------------------------------------+
    // |  HOG Feature Straight Forward Generator          |
    // +--------------------------------------------------+
    template <featEnum T=featType>
    static FeatImage<float> gen( cv::Mat &img, float y, float x,
                                 float rotation, float scale,
                                 int patch_stride = 3,
                                 int patch_size = 3,
                                 ENABLE_IF( HOG == T ) )
    {
      cv::Mat tmp;
      cv::resize( img, tmp, cv::Size( img.cols * scale, img.rows * scale ) );
      

      int radius = patch_size >> 1;
      
      float vert_y = patch_stride * sinf( rotation );
      float vert_x = patch_stride * cosf( rotation );
      float horz_y = vert_x;
      float horz_x = -vert_y;

      float y0 = y - vert_y * radius - horz_y * radius;
      flaot x0 = x - vert_x * radius - horz_x * radius;

      for ( int i=0; i<patch_size; i++ ) {
        int y1 = y0;
        int x1 = x0;
        for ( int j=0; j<patch_size; j++ ) {

          
          

          y1 += horz_y;
          x1 += horz_x;
        }
        y0 += vert_y;
        x0 += vert_x;
      }
      
    }
    
      


    // +--------------------------------------------------+
    // |  HOG Feature Generator                           |
    // +--------------------------------------------------+

    template <featEnum T=featType>
    static FeatImage<float> gen( std::string filename, ENABLE_IF( HOG == T ) )
    {
      cv::Mat raw = cv::imread( filename );
      if ( raw.empty() ) {
        Error( "cvFeat<HOG::gen()   Failed to load image %d", filename.c_str() );
        exit( -1 );
      }

      return gen<HOG>( raw );
    }


    // +--------------------------------------------------+
    // |  HOG Feature Generator with scale                |
    // +--------------------------------------------------+
    template <featEnum T=featType>
    static FeatImage<float> gen( std::string filename, int scales, float base, ENABLE_IF( HOG == T ) )
    {
      
      cv::Mat raw = cv::imread( filename );
      if ( raw.empty() ) {
        Error( "cvFeat<HOG::gen()   Failed to load image %d", filename.c_str() );
        exit( -1 );
      }

      std::vector<FeatImage<float> > imgs;

      imgs.resize( ( scales << 1 ) + 1 );
      
      imgs[scales] = std::move( gen<HOG>( raw ) );

      cv::Mat org = raw.clone();
      cv::Mat tmp;
      float scale = base;
      for ( int i=-1; i>=-scales; i-- ) {
        cv::resize( org, tmp, cv::Size( org.cols * scale, org.rows * scale ) );
        scale = scale * base;
        imgs[scales+i] = std::move( gen<HOG>( tmp ) );
      }


      
      scale = 1.0 / base;
      for ( int i=1; i<=scales; i++ ) {
        cv::resize( org, tmp, cv::Size( org.cols * scale, org.rows * scale ) );
        scale = scale / base;
        imgs[scales+i] = std::move( gen<HOG>( tmp ) );
      }
      
      return FeatImage<float>( std::move(imgs), base );
    }



    // +--------------------------------------------------+
    // |  HOG Feature Generator Core                      |
    // +--------------------------------------------------+
    
    template <featEnum T=featType>
    static FeatImage<float> gen( cv::Mat raw, ENABLE_IF( HOG == T ) )
    {
      FeatImage<float> img;
      
      /* Image Normalization */
      cv::Mat normalized;
      cv::normalize( raw, normalized, 0, 255, cv::NORM_MINMAX );


      /* gradient */
      cv::Mat gradx( raw.rows, raw.cols, CV_32FC1 );
      cv::Mat grady( raw.rows, raw.cols, CV_32FC1 );
      if ( options.enable_color && 3 == normalized.channels() ) {
        cv::Mat multiGradx, multiGrady;
        cv::Sobel( normalized, multiGradx, CV_32F, 1, 0 );
        cv::Sobel( normalized, multiGrady, CV_32F, 0, 1 );

        for ( int i=0; i<raw.rows; i++ ) {
          float *mxptr = multiGradx.ptr<float>(i);
          float *myptr = multiGrady.ptr<float>(i);
          float *gxptr = gradx.ptr<float>(i);
          float *gyptr = grady.ptr<float>(i);
          for ( int j=0; j<raw.cols; j++ ) {
            float max = (*mxptr) * (*mxptr) + (*myptr) * (*myptr);
            *gxptr = *(mxptr++);
            *gyptr = *(myptr++);
            for ( int k=1; k<3; k++ ) {
              float norm = (*mxptr) * (*mxptr) + (*myptr) * (*myptr);
              if ( norm > max ) {
                max = norm;
                *gxptr = *mxptr;
                *gyptr = *myptr;
              }
              mxptr++;
              myptr++;
            }
            gxptr++;
            gyptr++;
          }
        }
      } else if ( 1 == normalized.channels() ) {
        cv::Sobel( normalized, gradx, CV_32F, 1, 0 );
        cv::Sobel( normalized, grady, CV_32F, 0, 1 );
      }

      /* convert to rad and norm */
      cv::Mat radMat( raw.rows, raw.cols, CV_32FC1 );
      cv::Mat normMat( raw.rows, raw.cols, CV_32FC1 );

      for ( int i=0; i<raw.rows; i++ ) {
        float *gxptr = gradx.ptr<float>(i);
        float *gyptr = grady.ptr<float>(i);
        float *rad_ptr = radMat.ptr<float>(i);
        float *norm_ptr = normMat.ptr<float>(i);
        for ( int j=0; j<raw.cols; j++ ) {
          *(rad_ptr++) = atan2( *gyptr, *gxptr );
          *(norm_ptr++) = sqrt( (*gyptr) * (*gyptr) + (*gxptr) * (*gxptr) );
          gyptr++;
          gxptr++;
        }
      }

      /* initialization of voting */

      FeatImage<float> pointMap( raw.rows, raw.cols, options.orientation_bins );
      int lo_bin = 0;
      int hi_bin = 0;
      float lo_wt = 0.0f;
      float hi_wt = 0.0f;
      {
        int k = 0;
        for ( int i=0; i<raw.rows; i++ ) {
          float *rad_ptr = radMat.ptr<float>(i);
          float *norm_ptr = normMat.ptr<float>(i);
          for ( int j=0; j<raw.cols; j++ ) {
            Angle2Bin( *rad_ptr, lo_bin, hi_bin, lo_wt, hi_wt );

            pointMap[k][lo_bin] += lo_wt * (*norm_ptr);
            pointMap[k][hi_bin] += hi_wt * (*norm_ptr);
            rad_ptr++;
            norm_ptr++;
            k++;
          }
        }
      }


      /* voting */
      pointMap.MeanFilter( options.cell_side >> 1 );

      return pointMap;

    }
    
  };

  template <featEnum featType>
  class cvAlbum
  {
  public:
    static Album<typename GetDataType<featType>::type> gen( const std::vector<std::string>& lst )
    {
      Info( "Creating Album ..." );
      Album<typename GetDataType<featType>::type> album;
      for ( int i=0, end=static_cast<int>( lst.size() ); i<end; i++ ) {
        auto img = cvFeat<featType>::gen( lst[i] );
        Info( "%d/%d %s", i+1, end, lst[i].c_str() );
        album.push( std::move( img ) );
      }
      Done( "Album created." );
      return album;
    }
  };


  // +-------------------------------------------------------------------------------
  // | Image Display (with show patch functionality and callback)
  // +-------------------------------------------------------------------------------

  
  class ImageViewer
  {
  private:
    std::string wndName;
    cv::Mat image;
    int align;
    int radius;
    std::function<void(int,int)> callback;
  public:
    static const int CENTER = 0;
    static const int TOPLEFT = 1;
    
    ImageViewer( std::string wnd, cv::Mat img, int r=6 ) : align(CENTER)
    {
      wndName = wnd;
      image = img.clone();
      cv::imshow( wndName, image );
      cv::setMouseCallback( wndName, MouseCallback, this );
      callback = [](int x, int y){ Info( "(%d,%d) Clicked.", y, x ); };
      radius = r;
    }

    // ImageViewer( std::string wnd, int index, std::vector<std::string> &imgList )
    //   : align(CENTER) // index = image index in Dataset
    // {
    //   wndName = wnd;
    //   image = cv::imread( strf( "%s/%s.png", env["dataset"].c_str(), imgList[index].c_str() ) );
    //   display( -1, -1 );
    //   callback = [](int x, int y){ Info( "(%d,%d) Clicked.", y, x ); };
    // }

    
    void display( int x, int y )
    {
      cv::Mat canvas = image.clone();
      if ( -1 != x ) {
        if ( CENTER == align ) {
          rectangle( canvas,
                     cv::Point( x-radius, y-radius ),
                     cv::Point( x+radius, y+radius ),
                     cv::Scalar( 0, 255, 0 ) ) ;
        } else if ( TOPLEFT == align ) {
          rectangle( canvas,
                     cv::Point( x-radius, y-radius ),
                     cv::Point( x+radius, y+radius ),
                     cv::Scalar( 0, 255, 0 ) ) ;
        }
      }
      cv::imshow( wndName, canvas );
      cv::setMouseCallback( wndName, MouseCallback, this );
    }

    


    void display( const std::vector<PatLoc> &list )
    {
      cv::Mat canvas = image.clone();
      int side = ( ( radius << 1 ) + 1 );
      cv::Point2f vertices[4];
      for ( auto& ele : list ) {
        cv::RotatedRect( cv::Point2f( ele.x, ele.y ),
                         cv::Size2f( side * ele.scale, side * ele.scale ),
                         ele.rotation / M_PI * 180.0 ).points( vertices );
        for ( int i=0; i<4; i++ ) {
          cv::line( canvas, vertices[i], vertices[(i+1) & 3], cv::Scalar( 0, 255, 0 ) );
        }
      }
      cv::imshow( wndName, canvas );
      cv::setMouseCallback( wndName, MouseCallback, this );
    }

                  
    // void chImg( const int index, const std::vector<std::string> &imgList )
    // {
    //   image = cv::imread( strf( "%s/%s.png", env["dataset"].c_str(), imgList[index].c_str() ) );
    //   display( -1, -1 );
    // }

    void setCallback( std::function<void(int,int)> cb ) {
      callback = cb;
    }

    static void MouseCallback( int event, int x, int y, int __attribute__((__unused__)) flags, void *param )
    {
      if ( event == CV_EVENT_LBUTTONDOWN ) {
        ImageViewer *viewer = (ImageViewer*) param;
        viewer->display( x, y );
        viewer->callback( x, y );
      }
    }
  };




}

/* definitions for static members */
template <PatTk::featEnum featType>
PatTk::FeatOptions<featType> PatTk::cvFeat<featType>::options;





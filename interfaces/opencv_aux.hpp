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

  enum featEnum { BGR, Lab, HOG, DEFAULT_FEAT, BGR_FLOAT };
  
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
        img.ToggleNormalized( false );
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
        img.ToggleNormalized( false );
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
        Error( "cvFeat<BGR::gen()   Failed to load image %s", filename.c_str() );
        exit( -1 );
      }
      return gen<BGR>( raw );
    }


    template <featEnum T=featType>
    static FeatImage<uchar> gen( std::string filename, int scales, float base, ENABLE_IF( BGR == T ) )
    {
      
      cv::Mat raw = cv::imread( filename );
      if ( raw.empty() ) {
        Error( "cvFeat<BGR::gen()   Failed to load image %s", filename.c_str() );
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
    // |  BGR Feature Generator (float version)           |
    // +--------------------------------------------------+
    template <featEnum T=featType>
    static FeatImage<float> gen( cv::Mat raw, ENABLE_IF( BGR_FLOAT == T ) )
    {
      if ( 1 == raw.channels() ) {
        FeatImage<float> img( raw.rows, raw.cols, 3 );
        float *img_ptr = img[0];
        for ( int i=0; i<raw.rows; i++ ) {
          uchar *raw_ptr = raw.ptr<uchar>(i);
          for ( int j=0; j<raw.cols; j++ ) {
            *(img_ptr++) = static_cast<float>( *raw_ptr ) / 255.0f;
            *(img_ptr++) = static_cast<float>( *raw_ptr ) / 255.0f;
            *(img_ptr++) = static_cast<float>( *raw_ptr ) / 255.0f;
            raw_ptr++;
          }
        }
        img.ToggleNormalized( false );
        return img;
      } else if ( 3 == raw.channels() ) {
        FeatImage<float> img( raw.rows, raw.cols, 3 );
        float *img_ptr = img[0];
        size_t row_size = sizeof(uchar) * 3 * raw.cols;
        for ( int i=0; i<raw.rows; i++ ) {
          uchar *raw_ptr = raw.ptr<uchar>(i);
          for ( size_t j=0; j<row_size; j++ ) {
            *(img_ptr++) = static_cast<float>( *(raw_ptr++) ) / 255.0f;
          }
        }
        img.ToggleNormalized( false );
        return img;
      }

      Error( "cvFeat<BGR>::gen()   Bad number of channels: %d\n", raw.channels() );
      exit( -1 );
    }

    template <featEnum T=featType>
    static FeatImage<float> gen( std::string filename, ENABLE_IF( BGR_FLOAT == T ) )
    {
      cv::Mat raw = cv::imread( filename );
      if ( raw.empty() ) {
        Error( "cvFeat<BGR::gen()   Failed to load image %s", filename.c_str() );
        exit( -1 );
      }
      return gen<BGR_FLOAT>( raw );
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
    // |  HOG Feature Generator                           |
    // +--------------------------------------------------+

    template <featEnum T=featType>
    static FeatImage<float> gen( std::string filename, ENABLE_IF( HOG == T ) )
    {
      cv::Mat raw = cv::imread( filename );
      if ( raw.empty() ) {
        Error( "cvFeat<HOG::gen()   Failed to load image %s", filename.c_str() );
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
        Error( "cvFeat<HOG::gen()   Failed to load image %s", filename.c_str() );
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


      
      scale = 1.0f / base;
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

      pointMap.SetRotBins( options.orientation_bins );

      pointMap.InitInvNorms();

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
    

    void SwitchImage( std::string filename ) 
    {
      image = cv::imread( filename );
      if ( image.empty() ) {
        Error( "ImageViewer.SwitchImage(): error while reading image %s", filename.c_str() );
        exit( -1 );
      }
    }

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


  // +-------------------------------------------------------------------------------
  // |  Visualization sub-Library
  // +-------------------------------------------------------------------------------
  void put( const cv::Mat& icon, const int y, const int x, cv::Mat& bg, const int zoom = 1 )
  {
    if ( CV_8UC3 != icon.type() || CV_8UC3 != icon.type() ) {
      Error( "IconList::put() - either icon or bg is not of CV_8UC3" );
      exit( -1 );
    }
    for ( int dy=0; dy<icon.rows; dy++ ) {
      for ( int dx=0; dx<icon.cols; dx++ ) {
        for ( int k=0; k<3; k++ ) {
          for ( int zy=0; zy<zoom; zy++ ) {
            for ( int zx=0; zx<zoom; zx++ ) {
              bg.at<cv::Vec3b>( y + dy * zoom + zy, x + dx * zoom + zx )[k] = icon.at<cv::Vec3b>( dy, dx )[k];
            }
          }
        }
      }
    }
  }
  

  class IconList
  {
  private:
    std::vector<cv::Mat> icons;
    std::function<void(const int)> callback;
  public:
    std::vector<PatLoc> patches;
    std::string window;
    int patchSize;
    struct Options
    {
      int margin;
      int cols;
      int colSep;
      int rowSep;
      int zoom;
      cv::Scalar background;
    } options;
    
  private:
    IconList();

    static void MouseCallback( int event, int x, int y, int __attribute__((__unused__)) flags, void *param )
    {
      if ( event == CV_EVENT_LBUTTONDOWN ) {
        IconList *ptr = (IconList*) param;
        ptr->display( y, x );
      }
    }
    
  public:
    IconList( const std::string& win, int size ) : window(win), patchSize(size)
    {
      options.margin = 5;
      options.cols = 40;
      options.colSep = 5;
      options.rowSep = 5;
      options.zoom = 1;
      options.background = cv::Scalar( 0, 0, 0 );
      callback = [](const int index){ printf("%d clicked.\n", index ); };
    }

    ~IconList()
    {
      close();
    }
    
    void clear()
    {
      icons.clear();
      patches.clear();
    }


    void push( const std::vector<std::string>& imgList, const PatLoc& loc )
    {
      cv::Mat tmp = cv::imread( imgList[loc.id] );
      if ( tmp.empty() ) {
        Error( "IconList::push() - failed to load image %s.", imgList[loc.id].c_str() );
        exit( -1 );
      }
      if ( CV_8UC3 != tmp.type() ) {
        Error( "IconList::push() - image is not of type CV_8UC3." );
        exit( -1 );
      }


      auto img = cvFeat<BGR>::gen( tmp ); 
      img.SetPatchStride(1);
      img.SetPatchSize( patchSize );
      uchar patch[img.GetPatchDim()];
      img.FetchPatch( loc.y, loc.x, patch ); //loc.rotation, loc.scale, patch );


      icons.push_back( cv::Mat( patchSize, patchSize, CV_8UC3 ) );
      cv::Mat& icon = icons.back();


      int i = 0;
      for ( int dy=0; dy<patchSize; dy++ ) {
        for ( int dx=0; dx<patchSize; dx++ ) {
          for ( int k=0; k<3; k++ ) {
            icon.at<cv::Vec3b>( dy, dx )[k] = patch[i++];
          }
        }
      }

      patches.push_back( loc );
    }

    void push( const cv::Mat tmp, const PatLoc& loc )
    {

      if ( CV_8UC3 != tmp.type() ) {
        Error( "IconList::push() - image is not of type CV_8UC3." );
        exit( -1 );
      }

      
      loc.show();
      

      auto img = cvFeat<BGR>::gen( tmp );
      img.SetPatchStride(1);
      img.SetPatchSize( patchSize );
      uchar patch[img.GetPatchDim()];

      img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, patch );
      
      icons.push_back( cv::Mat( patchSize, patchSize, CV_8UC3 ) );
      cv::Mat& icon = icons.back();
      int i = 0;
      for ( int dy=0; dy<patchSize; dy++ ) {
        for ( int dx=0; dx<patchSize; dx++ ) {
          for ( int k=0; k<3; k++ ) {
            icon.at<cv::Vec3b>( dy, dx )[k] = patch[i++];
          }
        }
      }
      patches.push_back( loc );
    }

    
    
    void setCallback( std::function<void(const int)> cb ) {
      callback = cb;
    }

    // ( my, mx ) represents the position of the last mouse click.
    void display( const int my = -1, const int mx = -1 )
    {
      if ( 0 == icons.size() ) {
        Info( "IconList::display() - nothing to show." );
        return ;
      }
      int width = options.margin * 2 + options.colSep * ( options.cols - 1 )
        + icons[0].cols * options.zoom * options.cols;
      int rows = ( icons.size() - 1 ) / options.cols + 1;
      int height = options.margin * 2 + options.rowSep * ( rows - 1 ) + icons[0].rows * options.zoom * rows;
      cv::Mat canvas( height, width, CV_8UC3, options.background );

      int y = options.margin;
      int x = options.margin;
      for ( uint i=0; i<icons.size(); i++ ) {
        put( icons[i], y, x, canvas, options.zoom );
        if ( -1 != my && -1 != mx ) {
          // highlight
          if ( y <=  my && my < y + icons[0].rows * options.zoom &&
               x <= mx && mx < x + icons[0].cols * options.zoom ) {
            callback( i );
            rectangle( canvas,
                       cv::Point( x-2, y-2 ),
                       cv::Point( x + icons[0].cols * options.zoom + 1, y + icons[0].rows * options.zoom + 1 ),
                       cv::Scalar( 0, 255, 0 ) ) ;
          }
        }
        if ( 0 == (i+1) % options.cols ) {
          y += options.rowSep + icons[0].rows * options.zoom;
          x = options.margin;
        } else {
          x += options.colSep + icons[0].cols * options.zoom;
        }
      }
      cv::imshow( window, canvas );
      cv::setMouseCallback( window, MouseCallback, this );
    }

    void close()
    {
      cv::destroyWindow( window );
      cv::waitKey(1);
    }

  };
  
}

/* definitions for static members */
template <PatTk::featEnum featType>
PatTk::FeatOptions<featType> PatTk::cvFeat<featType>::options;





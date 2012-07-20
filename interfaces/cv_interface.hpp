/*********************************************************************************
 * File: cv_interface.hpp
 * Description: PatTk interfaces with open CV library
 * by BreakDS, University of Wisconsin Madison, Fri Jul 13 13:14:44 CDT 2012
 *********************************************************************************/


#pragma once
#include <string>
#include <functional>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "data/2d.hpp"
#include "data/features.hpp"
#include "opencv2/opencv.hpp"

using namespace EnvironmentVariable;

namespace PatTk
{


  // +-------------------------------------------------------------------------------
  // |  Image<cellType, typename valueType> Generators
  // +-------------------------------------------------------------------------------
  
  template <typename cellType, typename valueType>
  class cvFeatGen
  {
  public:
    static Image<cellType,valueType> gen( const cv::Mat __attribute__((__unused__)) &raw )
    {
      return Image<cellType,valueType>();
    }
  };
  
  
  /// LabCell Specialization
  template <typename valueType>
  class cvFeatGen<LabCell, valueType>
  {
  public:
    static Image<LabCell, valueType> gen( const cv::Mat& raw )
    {
      Image<LabCell, valueType > img( raw.rows, raw.cols );
      cv::Mat lab;
      cv::cvtColor( raw, lab, CV_BGR2Lab );

      int i = 0;
      for ( auto it = lab.begin<cv::Vec3b>(), end = lab.end<cv::Vec3b>(); it != end; it ++ ) {
        img.setCell( i, LabCell( (*it)[0], (*it)[1], (*it)[2] ) );
        i++;
      }
      return img;
    }
  };


  // HoGCell Specialization
  template <typename valueType>
  class cvFeatGen<HoGCell,valueType>
  {
  public:
    static Image<HoGCell,valueType> gen( const cv::Mat& raw )
    {
      static int hogBins = env["hog-bins"];
      cv::Mat lab = cv::Mat::zeros( raw.rows, raw.cols, CV_8UC3 );
      cv::cvtColor( raw, lab, CV_BGR2Lab );
      cv::Mat gray( raw.rows, raw.cols, CV_8UC1 );
      cv::Mat_<cv::Vec3b>::const_iterator it = lab.begin<cv::Vec3b>();
      cv::Mat_<cv::Vec3b>::const_iterator end = lab.end<cv::Vec3b>();
      cv::Mat_<uchar>::iterator grayit = gray.begin<uchar>();
      for ( ; it != end; it++ ) {
        *(grayit++) = (*it)[0];
      }
      
      cv::Mat gradx, grady;
      cv::Sobel( gray, gradx, CV_32F, 1, 0 );
      cv::Sobel( gray, grady, CV_32F, 0, 1 );
      
      Image<HistCell<double>, valueType > img( raw.rows, raw.cols );
      
      cv::Mat_<float>::iterator itGradx = gradx.begin<float>();
      cv::Mat_<float>::iterator itGrady = grady.begin<float>();
      cv::Mat_<float>::iterator itEnd = gradx.end<float>();

      double radCell = M_PI / hogBins;
      uint i = 0;
      for ( ; itGradx != itEnd; itGradx++, itGrady++ ){
        double dx = static_cast<double>( *itGradx );
        double dy = static_cast<double>( *itGrady );
        
        double val = sqrt( dx * dx + dy * dy );
        double rad = atan2( dy, dx );
        if ( rad < 0 ) {
          rad += M_PI;
        }
        double pos = rad / radCell - 0.5;
        int b0 = static_cast<int>( floor( pos ) );
        int b1 = b0 + 1;
        double w0 = b1 - pos;
        double w1 = 1.0 - w0;
        if ( b0 < 0) b0 = hogBins - 1;
        if ( b1 >= hogBins ) b1 -= hogBins;
        img[i].reset( hogBins );
        
        img[i][b0] = w0 * val;
        img[i][b1] = w1 * val;
        i++;
      }

      IntegralImage( img, env["hog-cell-size"] );

      Image<HoGCell, valueType > hog( raw.rows, raw.cols );
      for ( int i=0,end=raw.rows*raw.cols; i<end; i++ ) {
        hog[i].reset( hogBins );
        img[i].NormalizeToUchar( hog[i] );
      }
      return hog;
    }
  };

  // +-------------------------------------------------------------------------------
  // |  Album<cellType, typename valueType> Generators
  // +-------------------------------------------------------------------------------
  template <typename cellType, typename valueType>
  class cvAlbumGen
  {
  public:
    static Album<cellType,valueType> gen( const vector<std::string>& lst )
    {
      Info( "Creating Album..." );
      Album<cellType,valueType> album;
      for ( int i=0, end=lst.size(); i<end; i++ ) {
        cv::Mat raw = cv::imread( lst[i] );
        album.push( std::move( cvFeatGen<cellType,valueType>::gen(raw) ) );
        album.back().setFullPath( lst[i] );
      }
      Done( "Album created." );
      return album;
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

  template <typename patchType>
  class IconList
  {
  private:
    vector<cv::Mat> icons;
    
    std::function<void(const int)> callback;
  public:
    vector<patchType> patches;
    std::string window;
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

    IconList( const std::string& win ) : window(win)
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
    
    void push( const patchType& patch )
    {
      cv::Mat tmp = cv::imread( patch.parent.fullpath );
      if ( tmp.empty() ) {
        Error( "IconList::push() - failed to load image." );
        exit( -1 );
      }
      if ( CV_8UC3 != tmp.type() ) {
        Error( "IconList::push() - image is not of type CV_8UC3." );
        exit( -1 );
      }
      icons.push_back( cv::Mat( patch.parent.GetPatchHeight(), patch.parent.GetPatchWidth(), CV_8UC3 ) );
      cv::Mat& icon = icons.back();
      for ( int dy=0; dy<patch.parent.GetPatchHeight(); dy++ ) {
        for ( int dx=0; dx<patch.parent.GetPatchWidth(); dx++ ) {
          for ( int k=0; k<3; k++ ) {
            icon.at<cv::Vec3b>( dy, dx )[k] = tmp.at<cv::Vec3b>( patch.y + dy, patch.x + dx )[k];
          }
        }
      }
      
      patches.push_back( patch );
    }

    void setCallback( std::function<void(const int)> cb ) {
      callback = cb;
    }

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
        // highlight
        if ( -1 != my && -1 != mx ) {
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


};

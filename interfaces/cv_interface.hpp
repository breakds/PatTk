/*********************************************************************************
 * File: cv_interface.hpp
 * Description: PatTk interfaces with open CV library
 * by BreakDS, University of Wisconsin Madison, Fri Jul 13 13:14:44 CDT 2012
 *********************************************************************************/


#pragma once
#include <string>
#include <functional>
#include "LLPack/utils/extio.hpp"
#include "data/2d.hpp"
#include "data/features.hpp"
#include "opencv2/opencv.hpp"


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
      cvtColor( raw, lab, CV_BGR2Lab );

      int i = 0;
      for ( auto it = lab.begin<cv::Vec3b>(), end = lab.end<cv::Vec3b>(); it != end; it ++ ) {
        img.setCell( i, LabCell( (*it)[0], (*it)[1], (*it)[2] ) );
        i++;
      }
      return img;
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
        album.push( cvFeatGen<cellType,valueType>::gen(raw) );
      }
      Done( "Album created." );
      return album;
    }
  };

  // +-------------------------------------------------------------------------------
  // |  Visualization sub-Library
  // +-------------------------------------------------------------------------------
  void put( const cv::Mat& icon, const int y, const int x, cv::Mat& bg )
  {
    if ( CV_8UC3 != icon.type() || CV_8UC3 != icon.type() ) {
      Error( "IconList::put() - either icon or bg is not of CV_8UC3" );
      exit( -1 );
    }
    for ( int dy=0; dy<icon.rows; dy++ ) {
      for ( int dx=0; dx<icon.cols; dx++ ) {
        for ( int k=0; k<3; k++ ) {
          bg.at<cv::Vec3b>( y + dy, x + dx )[k] = icon.at<cv::Vec3b>( y, x )[k];
        }
      }
    }
  }

  
  class IconList
  {
  private:
    vector<cv::Mat> icons;
    std::function<void(const int)> callback;
  public:
    std::string window;
    struct Options
    {
      int margin;
      int cols;
      int colSep;
      int rowSep;
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
      options.background = cv::Scalar( 0, 0, 0 );
      callback = [](const int index){ printf("%d clicked.\n", index ); };
    }

    ~IconList()
    {
      close();
    }

    
    template <typename patchType>
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
      tmp.release();
    }

    void display( const int my = -1, const int mx = -1 )
    {
      if ( 0 == icons.size() ) {
        Info( "IconList::display() - nothing to show." );
        return ;
      }
      int width = options.margin * 2 + options.colSep * ( options.cols - 1 ) + icons[0].cols * options.cols;
      int rows = ( icons.size() - 1 ) / options.cols + 1;
      int height = options.margin * 2 + options.rowSep * ( rows - 1 ) + icons[0].rows * rows;
      cv::Mat canvas( height, width, CV_8UC3, options.background );

      int y = options.margin;
      int x = options.margin;
      for ( uint i=0; i<icons.size(); i++ ) {
        put( icons[i], y, x, canvas );
        // highlight
        if ( -1 != my && -1 != mx ) {
          if ( y <=  my && my < y + icons[0].rows &&
               x <= mx && mx < x + icons[0].cols ) {
            callback( i );
            rectangle( canvas,
                       cv::Point( x-2, y-2 ),
                       cv::Point( x + icons[0].cols + 1, y + icons[0].rows + 1 ),
                       cv:: Scalar( 0, 255, 0 ) ) ;
          }
        }
        if ( 0 == (i+1) % options.cols ) {
          y += options.rowSep + icons[0].rows;
          x = options.margin;
        } else {
          x += options.colSep + icons[0].cols;
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

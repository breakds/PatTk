/*********************************************************************************
 * File: FakeLabel.cpp
 * Description: For Comparison of the Candidate Set retrieved by querying the tree
 *              and by applying fake label optimization
 * by BreakDS, University of Wisconsin Madison, Tue Jul 24 13:30:35 CDT 2012
 *********************************************************************************/

#include <type_traits>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/pathname.hpp"
#include "LLPack/algorithms/random.hpp"
#include "LLPack/utils/Environment.hpp"
#include "data/metric.hpp"
#include "query/tree.hpp"
#include "query/kernel.hpp"
#include "interfaces/cv_interface.hpp"

#define CELL HoGCell


using namespace EnvironmentVariable;
using namespace PatTk;


struct MCP
{
  cv::Mat& testImg;
  Image<CELL,int>& target;
  IconList<Image<CELL,int>::Patch>& dbg;
  IconList<Image<CELL,int>::Patch>& topRe;
  Tree<BasicKernel<CELL,int> >& tree;
  vector<vector<vector<Image<CELL,int>::Patch> > > &candidates;
  vector< vector<int> > &best;
};



int main( int argc, char **argv )
{
  srand( 1342464046 );

  if ( argc < 2 ) {
    Error( "Please specify a configuration file." );
    exit( -1 );
  }

  env.parse( argv[1] );
  env.Summary();


  // 1. Read Images
  auto album = cvAlbumGen<CELL,int>::gen( path::FFFL( env["folder"], env["files"], ".png" ) );
  album.SetPatchParameter( env["patch-size"], env["patch-size"], env["cell-stride"] );
  Done( "%d images loaded.", album.size() );

  // 2. Read Trees
  Tree<BasicKernel<CELL,int> > tree;
  Info( "Loading trees ..." );
  tree.read( strf( "%s/%s.dat", env["folder"].c_str(), env["tree-prefix"].c_str() ), album );
  Done( "Tree loaded.\n" );


  // 3. Read Target Image
  cv::Mat testImg = cv::imread( strf( "%s/%s.png", env["folder"].c_str(), env["target"].c_str() ) );
  auto target = cvFeatGen<CELL,int>::gen( testImg );
  target.SetPatchParameter( env["patch-size"], env["patch-size"], env["cell-stride"] );
  
  // 4. Query for preliminary Candidates
  vector< vector< vector<Image<CELL,int>::Patch> > > candidates;
  candidates.resize( target.rows );
  for ( int i=0; i<target.rows; i++ ) {
    candidates[i].resize( target.cols );
    for ( int j=0; j<target.cols; j++ ) {
      auto patch = target.Spawn( i, j );
      if ( patch.isValid() ) {
        auto node = tree.direct( patch );
        candidates[i][j].clear();
        for ( auto& cand : node->patches ) {
          candidates[i][j].push_back( cand );
          // Debugging:
          //          printf( "%d\n", candidates[i][j].back().parent.cols );
        }
      }
      if ( 0 == ( i * target.cols + j ) % 100 ) {
        Info( "[%3d\%] finished for query.", ( i * target.cols + j ) * 100 / ( target.rows * target.cols ) );
      }
    }
  }
  Done( "Preliminary query completed." );


  // 5. Refining Optimization (Experimental Codes)
  int maxIter = env["max-iter"];


  vector< vector<int> > best;
  best.resize( target.rows );
  for ( int i=0; i<target.rows; i++ ) {
    best[i].resize( target.cols );
    for ( int j=0; j<target.cols; j++ ) {
      best[i][j] = 0;
    }
  }
  for ( int iter=0; iter<maxIter; iter++ ) {
    // iteratively update best candidate for
    double energy = 0.0;
    if ( 0 == ( iter & 1 ) ) {
      for ( int i=0; i<target.rows; i++ ) {
        for ( int j=0; j<target.cols; j++ ) {
          if ( candidates[i][j].empty() ) continue;
          double minimum = 0;
          for ( int k=0, end=static_cast<int>( candidates[i][j].size() ); k<end; k++ ) {
            double val = 0;
            if ( i > 0 && !candidates[i-1][j].empty() ) {
              double tmp = FakeLabelDist<CELL,int>( candidates[i-1][j][best[i-1][j]],
                                                    candidates[i][j][k] );
              val += tmp * tmp;
            }
            if ( j > 0 && !candidates[i][j-1].empty() ) {
              double tmp = FakeLabelDist<CELL,int>( candidates[i][j-1][best[i][j-1]],
                                                    candidates[i][j][k] );
              val += tmp * tmp;
            }
            if ( i + 1 < target.rows && !candidates[i+1][j].empty() ) {
              double tmp = FakeLabelDist<CELL,int>( candidates[i+1][j][best[i+1][j]],
                                                    candidates[i][j][k] );
              val += tmp * tmp;
            }
            if ( j + 1 < target.cols && !candidates[i][j+1].empty() ) {
              double tmp = FakeLabelDist<CELL,int>( candidates[i][j+1][best[i][j+1]],
                                                    candidates[i][j][k] );
              val += tmp * tmp;
            }
            if ( 0 == k || val < minimum ) {
              minimum = val;
              best[i][j] = k;
            }

            if ( 0 == k || val < minimum ) {
              minimum = val;
              best[i][j] = k;
            }
          } // end for k
          energy += minimum;
        }
      }
    } else {
      for ( int i=0; i<target.rows; i++ ) {
        for ( int j=0; j<target.cols; j++ ) {
          if ( candidates[i][j].empty() ) continue;
          double minimum = 0;
          for ( int k=0, end=static_cast<int>( candidates[i][j].size() ); k<end; k++ ) {
            double val = 0;
            if ( i > 0 && !candidates[i-1][j].empty() ) {
              double tmp = FakeLabelDist<CELL,int>( candidates[i-1][j][best[i-1][j]],
                                                    candidates[i][j][k] );
              val += tmp * tmp;
            }
            if ( j > 0 && !candidates[i][j-1].empty() ) {
              double tmp = FakeLabelDist<CELL,int>( candidates[i][j-1][best[i][j-1]],
                                                    candidates[i][j][k] );
              val += tmp * tmp;
            }
            if ( i + 1 < target.rows && !candidates[i+1][j].empty() ) {
              double tmp = FakeLabelDist<CELL,int>( candidates[i+1][j][best[i+1][j]],
                                                    candidates[i][j][k] );
              val += tmp * tmp;
            }
            if ( j + 1 < target.cols && !candidates[i][j+1].empty() ) {
              double tmp = FakeLabelDist<CELL,int>( candidates[i][j+1][best[i][j+1]],
                                                    candidates[i][j][k] );
              val += tmp * tmp;
            }
            if ( 0 == k || val < minimum ) {
              minimum = val;
              best[i][j] = k;
            }
          } // end for k
          energy += minimum;
        }
      }
    }
    printf( "energy: %.5lf\n", energy );
    Info( "iter %d finished.", iter );
  }

  // display
  cv::imshow( "show", testImg );





  IconList<Image<CELL,int>::Patch> dbg( "debug" );
  IconList<Image<CELL,int>::Patch> topRe( "top" );
  MCP mcp = { testImg, target, dbg, topRe, tree, candidates, best };
  
  
  cv::setMouseCallback( "show",
                        []( int event, int x, int y,
                            int __attribute__((__unused__)) flags, void *param )
                        {
                          if ( CV_EVENT_LBUTTONDOWN == event) {
                            MCP *mcp = (MCP*) param;

                            int width = mcp->target.GetPatchWidth();
                            int height = mcp->target.GetPatchHeight();

                            int cx = x - ( width >> 1 );
                            int cy = y - ( height >> 1 );


                            if ( 0 <= cx && cx + width <= mcp->testImg.cols &&
                                 0 <= cy && cy + height <= mcp->testImg.rows ) {
                              cv::Mat canvas = mcp->testImg.clone();
                              rectangle( canvas,
                                         cv::Point( cx, cy ),
                                         cv::Point( cx + width, cy + height ),
                                         cv::Scalar( 0, 255, 0 ) ) ;
                              cv::imshow( "show", canvas );

                              mcp->dbg.clear();
                              mcp->dbg.options.zoom = 2;
                              for ( int k=0, end=mcp->candidates[cy][cx].size(); k<end; k++ ) {
                                mcp->dbg.push( mcp->candidates[cy][cx][k] );
                              }
                              mcp->dbg.display();

                              
                              mcp->topRe.clear();
                              mcp->topRe.options.zoom = 2;
                              mcp->topRe.push( mcp->candidates[cy][cx][mcp->best[cy][cx]] );
                              mcp->topRe.display();

                            }
                          }
                        }, &mcp );

  while ( 27 != cv::waitKey(30) ) ;
  

  
  
  
  return 0;
}


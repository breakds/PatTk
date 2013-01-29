#include "LLPack/utils/Environment.hpp"
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/time.hpp"
#include "LLPack/utils/pathname.hpp"
#include "LLPack/algorithms/random.hpp"
#include "../data/FeatImage.hpp"
#include "../data/Label.hpp"
#include "../interfaces/opencv_aux.hpp"
#include "../query/forest.hpp"

using namespace EnvironmentVariable;
using namespace PatTk;
using rndgen::randperm;


class VoteMap
{
  int rows, cols;
  std::vector<float> mat;
public:

  VoteMap( int r, int c ) : rows(r), cols(c)
  {
    mat.resize( r * c * LabelSet::classes );
    memset( &mat[0], 0, sizeof(float) * r * c * LabelSet::classes );
  }

  template <typename T>
  inline void vote( int y, int x, const T* votes )
  {
    float *p = &mat[ ( y * cols + x ) * LabelSet::classes ];
    const T *vp = votes;
    for ( int i=0; i<LabelSet::classes; i++ ) {
      *(p++) += static_cast<float>( *(vp++) );
    }
  }

  inline int getClass( int index ) const
  {
    int maxp = 0;
    const float *p = &mat[ index * LabelSet::classes ];
    for ( int i=1; i<LabelSet::classes; i++ ) {
      if ( p[i] > p[maxp] ) maxp = i;
    }
    return maxp;
  }

  inline int getClass( int y, int x ) const
  {
    return getClass( y * cols + x );
  }


  inline double cost( int index, int truth ) const
  {
    int winner = getClass( index );
    if ( winner == truth ) {
      return 0.0;
    } else {
      const float *p = &mat[ index * LabelSet::classes ];
      double ratio = p[truth] / p[winner];
      if ( ratio > 0.7 ) return ratio;
    }
    return 1.0;
  }


  inline double cost( int y, int x, int truth ) const
  {
    return cost( y * cols + x, truth );
  }

  inline double compare( const FeatImage<uchar>& truth )
  {
    double s = 0.0;
    for ( int i=0; i<cols*rows; i++ ) {
      s += cost( i, LabelSet::GetClass( truth(i)[0], truth(i)[1], truth(i)[2] ) );
    }
    return s;
  }

  cv::Mat synthesis() const
  {
    cv::Mat canvas( rows, cols, CV_8UC3 );
    for ( int y=0; y<rows; y++ ) {
      for ( int x=0; x<cols; x++ ) {
        auto color = LabelSet::GetColor( getClass( y, x ) );
        canvas.at<cv::Vec3b>( y, x )[0] = get<0>( color );
        canvas.at<cv::Vec3b>( y, x )[1] = get<1>( color );
        canvas.at<cv::Vec3b>( y, x )[2] = get<2>( color );
      }
    }
    return canvas;
  }
};
  

int main( int argc, char **argv )
{

  if ( argc < 2 ) {
    Error( "Missing configuration file in arguments. (treeQ.conf)" );
    exit( -1 );
  }

  // srand( 7325273 );
  srand(time(NULL));

  env.parse( argv[1] );
  env.Summary();

  LabelSet::initialize( env["color-map"] );

  /* ---------- Build/Load Forest ---------- */
  std::vector<std::string> imgList = std::move( readlines( strf( "%s/%s", env["dataset"].c_str(),
                                                                 env["list-file"].c_str())) );
  auto lblList = std::move( path::FFFL( env["dataset"], imgList, "_L.png" ) );
  imgList =  std::move( path::FFFL( env["dataset"], imgList, ".png" ) );

  Album<float> album;
  {
    int i = 0;
    int n = static_cast<int>( imgList.size() );
    for ( auto& ele : imgList ) {
      album.push( std::move( cvFeat<HOG>::gen( ele ) ) );
      progress( ++i, n, "Loading Album" );
    }
  }
  printf( "\n" );


  Album<uchar> lblAlbum;
  {
    int i = 0;
    int n = static_cast<int>( lblList.size() );
    for ( auto& ele : lblList ) {
      lblAlbum.push( std::move( cvFeat<BGR>::gen( ele ) ) );
      progress( ++i, n, "Loading Label Album" );
    }
  }
  printf( "\n" );
  lblAlbum.SetPatchSize( env["lbl-size"] );
  lblAlbum.SetPatchStride( 1 );
  

  
  /* ---------- Load Forest ---------- */
  Info( "Loading Forest .." );
  timer::tic();
  Forest<SimpleKernel<float> > forest( env["forest-dir"] );
  printf( "tree loaded: %.3lf sec\n", timer::utoc() );
  printf( "maxDepth: %d\n", forest.maxDepth() );

  
  /* ---------- Collective Entropy ---------- */
  if ( env.find( "entropy-output" ) ) {
    WITH_OPEN( out, env["entropy-output"].c_str(), "w" );
    uchar label[lblAlbum(0).GetPatchDim()];
    int count[LabelSet::classes];
    for ( int i=0; i<forest.centers(); i++ ) {
      memset( count, 0, sizeof(int) * LabelSet::classes );
      for ( auto& ele : forest(i).store ) {
        lblAlbum(ele.id).FetchPatch( ele.y, ele.x, label );
        for ( int j=0; j<lblAlbum(0).GetPatchDim(); j+=3 ) {
          count[LabelSet::GetClass( label[j], label[j+1], label[j+2] )]++;
        }
      }
      double ent = entropy( count, LabelSet::classes );
      fprintf( out, "%.8lf\n", ent );
      if ( 0 == i % 100 ) progress( i+1, forest.centers(), "Calculating Entropy" );
    }
    printf( "\n" );
    END_WITH( out );
  }


  /* ---------- Center Entropy ---------- */
  if ( env.find( "center-entropy-output" ) ) {
    WITH_OPEN( out, env["center-entropy-output"].c_str(), "w" );
    int count[LabelSet::classes];
    for ( int i=0; i<forest.centers(); i++ ) {
      memset( count, 0, sizeof(int) * LabelSet::classes );
      for ( auto& ele : forest(i).store ) {
        const uchar *color = lblAlbum(ele.id)(ele.y, ele.x);
        count[LabelSet::GetClass( color[0], color[1], color[2] )]++;
      }
      double ent = entropy( count, LabelSet::classes );
      fprintf( out, "%.8lf\n", ent );
      if ( 0 == i % 100 ) progress( i+1, forest.centers(), "Calculating Entropy" );
    }
    printf( "\n" );
    END_WITH( out );
  }


  /* ---------- Voting Test ---------- */
  if ( env.find( "reconstruct-output" ) ) {

    uchar label[lblAlbum(0).GetPatchDim()];
    
    // Build voters
    std::vector<std::vector<float> > voters;

    int voterSize = env["lbl-size"];
    
    voters.resize( forest.centers() );
    for ( int leafID=0; leafID<forest.centers(); leafID++ ) {
      voters[leafID].resize( LabelSet::classes * voterSize * voterSize );
      for ( auto& loc : forest(leafID).store ) {
        lblAlbum(loc.id).FetchPatch( loc.y, loc.x, label );
        for ( int i=0; i<voterSize*voterSize; i++ ) {
          int k = LabelSet::GetClass( label[i*3], label[i*3+1], label[i*3+2] );
          voters[leafID][ i * LabelSet::classes + k ] += 1.0;
        }
      }
      for ( int i=0; i<voterSize*voterSize*LabelSet::classes; i+=LabelSet::classes ) {
        float s = sum_vec( &voters[leafID][i], LabelSet::classes );
        scale( &voters[leafID][i], LabelSet::classes, 1.0f / s );
      }
    }
    
    
    int voterRadius = voterSize >> 1;
    float feat[album(0).GetPatchDim()];
    for ( int i=0; i<1; i++ ) {
      VoteMap voteMap( album(i).rows, album(i).cols );
      for ( int y=0; y<album(i).rows; y++ ) {
        for ( int x=0; x<album(i).cols; x++ ) {
          album(i).FetchPatch( y, x, feat );
          std::vector<int> res = std::move( forest.query( feat ) );
          for ( auto& leafID : res ) {
            int j = 0;
            for ( int dy=-voterRadius; dy<=voterRadius; dy++ ) {
              int y1 = y + dy;
              if ( 0 > y1 || album(i).rows <= y1 ) continue;
              for ( int dx=-voterRadius; dx<=voterRadius; dx++ ) {
                int x1 = x + dx;
                if ( 0 > x1 || album(i).cols <= x1 ) continue;
                voteMap.vote( y1, x1, &voters[leafID][j] );
                j += LabelSet::classes;
              } // for dx
            } // for dy
          } // for leafID
        } // for x
      } // for y

      WITH_OPEN( out, strf( "%s/%s.txt", env["reconstruct-output"].c_str(), imgList[i].c_str() ).c_str(), "w" );
      fprintf( out, "%.6lf\n", voteMap.compare( lblAlbum(i) ) );
      END_WITH( out );

      cv::Mat syn = voteMap.synthesis();
      cv::imwrite( strf( "%s/%s", env["reconstruct-output"].c_str(), imgList[i].c_str() ), syn );
      
    } // for i
    
  }

  

  
  return 0;
}



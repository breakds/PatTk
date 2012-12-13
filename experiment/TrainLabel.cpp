#include <vector>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/time.hpp"
#include "LLPack/utils/SafeOP.hpp"
#include "LLPack/utils/pathname.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/algorithms/heap.hpp"
#include "../data/FeatImage.hpp"
#include "../data/vector.hpp"
#include "../interfaces/opencv_aux.hpp"
#include "../query/forest.hpp"
#include "../graph/Trans.hpp"
#include "../graph/Bipartite.hpp"



using namespace PatTk;
using namespace EnvironmentVariable;



class Solver {
public:
  struct Options
  {
    double beta; // coefficient for the pairwise regularization term
    int maxIter; // iteration number
    double shrinkRatio; // Shrinking Ratio for line search
    double wolf; // Sufficent descreasing condition coefficient

    Options()
    {
      beta = 1.0;
      maxIter = 10;
      shrinkRatio = 0.8;
      wolf = 0.999;
    }
  } options;

private:
  double *D;
  const double *P;
  double *q;
  int M, L;
  const Forest<SimpleKernel<float> > *forest;
  const Bipartite *m_to_l;
    


  // D(m) = sum alpha(l,m)*q(l,m) - P(m)
  inline void update_D( int m )
  {

    auto& _to_l = m_to_l->getToSet( m );

    

    double *t = D + m * LabelSet::classes;
    memset( t, 0, sizeof(double) * LabelSet::classes );

    for ( auto& ele : _to_l ) {
      int l = ele.first;
      double alpha = ele.second;
      addScaledTo( t, q + l * LabelSet::classes, LabelSet::classes, alpha );
    }
    
    minusFrom( t, P + m * LabelSet::classes, LabelSet::classes );
  }


  inline double total_energy()
  {

    double energy = 0.0;
    double *Dp = D;
    for ( int m=0; m<M; m++ ) {
      energy += norm2( Dp, LabelSet::classes );
      Dp += LabelSet::classes;
    }


    double t0[LabelSet::classes];
    
    for ( int l=0; l<L; l++ ) {
      auto& _to_j = forest->GetWeights( l );
      for ( auto& ele : _to_j ) {
        int j = ele.first;
        if ( j<l ) {
          double wt = static_cast<double>( ele.second );
          minus( q + l * LabelSet::classes, q + j * LabelSet::classes, t0, LabelSet::classes );
          energy += options.beta * wt * norm2( t0, LabelSet::classes );
        }
      }
    }
    
    return energy;
    
  }


  /*
   * Restricted Energy on q(l) is
   *     sum_m ( D(m) - alpha(l) * q(l) + alpha(l) * q'(l) )^2
   *   + sum_j w(i,j) * ( q'(l) - q(j) )^2
   */
  inline double restrict_energy( int l, double *q_l = nullptr )
  {

    auto& _to_m = m_to_l->getFromSet( l );
    auto& _to_j = forest->GetWeights( l );
    
    double energy = 0.0;
    double t0[LabelSet::classes];

    if ( nullptr == q_l ) {
      for ( auto& ele : _to_m ) {
        int m = ele.first;
        energy += norm2( D + m * LabelSet::classes, LabelSet::classes );
      }

      for ( auto& ele : _to_j ) {
        int j = ele.first;
        double wt = static_cast<double>( ele.second );

        minus( q + l * LabelSet::classes, q + j * LabelSet::classes, t0, LabelSet::classes );
        energy += options.beta * wt * norm2( t0, LabelSet::classes );
      }
    } else {

      double t0[LabelSet::classes];

      for ( auto& ele : _to_m ) {
        int m = ele.first;
        double alpha = ele.second;

        memcpy( t0, D + m * LabelSet::classes, sizeof(double) * LabelSet::classes );
        minusScaledFrom( t0, q + l * LabelSet::classes, LabelSet::classes, alpha );
        addScaledTo( t0, q_l, LabelSet::classes, alpha );

        energy += norm2( t0, LabelSet::classes );
      }

      
      for ( auto& ele : _to_j ) {
        int j = ele.first;
        double wt = static_cast<double>( ele.second );
        minus( q_l, q + j * LabelSet::classes, t0, LabelSet::classes );
        energy += options.beta * wt * norm2( t0, LabelSet::classes );
      }

    }

    return energy;

  }

  inline void update_q( int l )
  {

    auto& _to_m = m_to_l->getFromSet( l );
    auto& _to_j = forest->GetWeights( l );
    
    double t0[LabelSet::classes];
    memset( t0, 0, sizeof(double) * LabelSet::classes );
    double t1[LabelSet::classes];
        
    
    for ( auto& ele : _to_m ) {
      int m = ele.first;
      double alpha = ele.second;
      addScaledTo( t0, D + m * LabelSet::classes, LabelSet::classes, alpha );
    }


    for ( auto& ele : _to_j ) {
      int j = ele.first;
      double wt = static_cast<double>( ele.second );
      minus( q + l * LabelSet::classes, q + j * LabelSet::classes, t1, LabelSet::classes );
      addScaledTo( t0, t1, LabelSet::classes, wt );
    }
     
    // Line Search
    double energy_old = restrict_energy( l ) * options.wolf;

    bool updated = false;

    
    for ( int i=0; i<40; i++ ) {
      scale( t0, LabelSet::classes, options.shrinkRatio );
      add( t0, q + l * LabelSet::classes, t1, LabelSet::classes );
      double energy_new = restrict_energy( l, t1 );

      if ( energy_new < energy_old ) {
        updated = true;
        break;
      }
    }

    if ( updated ) {
      // Simplex Projection
      watershed( t1, q + l * LabelSet::classes, LabelSet::classes );
    }
    
  }


public:

  Solver()
  {
    D = nullptr;
    P = nullptr;
    q = nullptr;
    M = 0;
    L = 0;
    forest = nullptr;
    m_to_l = nullptr;
  }
  
  /*
   * M = number of total patches
   * L = number of templates
   * forest = the forest
   * leafID = vector of leaf IDs xM
   * alpha =  vector of weights xM
   * P = label probability vector xM
   * q = label probability vector xL (solution)
   */
  void solve( int M1, int L1, const Forest<SimpleKernel<float> > *forest1,
              const Bipartite *m_to_l1,
              const double *P1, double *q1 )
  {

    M = M1;
    L = L1;
    D = new double[M*LabelSet::classes];
    P = P1;
    q = q1;
    forest = forest1;
    m_to_l = m_to_l1;


    
    for ( int iter=0; iter<options.maxIter; iter++ ) {
      // Update D(m) = sum alpha(l,m)*q(l,m) - P(m)
      for ( int m=0; m<M; m++ ) {
        update_D(m);
      }
      // Update q(l)'s
      for ( int l=0; l<L; l++ ) {
        update_q(l);
      }

      Info( "Iteration %d - Energy: %.5lf\n", iter, total_energy() );
      
    }

    DeleteToNullWithTestArray( D );
    
  }
  
};


int main( int argc, char **argv )
{

  if ( argc < 2 ) {
    Error( "Missing configuration file in options. (testing.conf)" );
    exit( -1 );
  }

  srand( 345645631 );
  

  env.parse( argv[1] );
  env.Summary();

  LabelSet::initialize( env["color-map"] );


  std::vector<std::string> nameList = std::move( readlines( strf( "%s/%s", env["dataset"].c_str(),
                                                                  env["list-file"].c_str())) );
  std::vector<std::string> imgList = std::move( path::FFFL( env["dataset"], nameList, ".png" ) );
  std::vector<std::string> labelList = std::move( path::FFFL( env["dataset"], nameList, "_L.png" ) );  
  
  for ( auto& ele : imgList ) {
    printf( "%s\n", ele.c_str() );
  }


  Info( "Loading Learning Album ..." );
  Album<float> album;
  for ( auto& ele : imgList ) {
    album.push( std::move( cvFeat<HOG>::gen( ele ) ) );
  }


  Album<uchar> labelAlbum;
  for ( auto& ele : labelList ) {
    labelAlbum.push( std::move( cvFeat<BGR>::gen( ele ) ) );
  }
  Done( "Learning Album Loaded" );
  
  
  Info( "Loading Forest ..." );
  Forest<SimpleKernel<float> > forest( env["forest-name"] );
  Done( "Forest Loaded." );
  
  Info( "Start Label Training ..." );



  /// Construct Bipartite Graph between l and m
  /// and also the ground truth P

  Info( "Initializing Optimization ... " );

  // calculate M
  int M = 0;
  for ( auto& img :album ) {
    M += ( img.rows - 14 ) * ( img.cols - 14 );
  }

  Bipartite m_to_l( M, forest.centers() );
  
  double *P = new double[M * LabelSet::classes];
  double *pP = P;

  int m = 0;
  for ( auto& img : album ) {
    printf( "working on Image %d ...\n", img.id );
    float feat[img.GetPatchDim()];
    for ( int i=7; i<img.rows-7; i++ ) {
      for ( int j=7; j<img.cols-7; j++ ) {
        img.FetchPatch( i, j, feat );
        auto res = std::move( forest.query_with_coef( feat ) );

        int count = 0;
        for ( auto& ele : res ) {
          if ( count++ > 100 ) break;
          m_to_l.add( m, ele.first, ele.second );
        }

        const uchar* color = labelAlbum(img.id)( i, j );
        int classID = LabelSet::GetClass( color[0],
                                          color[1],
                                          color[2] );

        for ( int k=0; k<LabelSet::classes; k++ ) {
          if ( k == classID ) {
            *(pP++) = 1.0;
          } else {
            *(pP++) = 0.0;
          }
        }
        m++;
      }
    }
  }
  Done( "Initialized." ); 
  
  double *q = new double[ forest.centers() * LabelSet::classes ];
  double *qp = q;
  for ( int i=0; i<forest.centers() * LabelSet::classes; i++ ) *(qp++) = LabelSet::inv;


  
  Solver solver;
  solver.options.beta = 0.0;
  Info( "Solving ..." );
  solver.solve( M, forest.centers(), &forest, &m_to_l, P, q );
  Done( "Solved." );


  DeleteToNullWithTestArray( q );
  DeleteToNullWithTestArray( P );
  return 0;
}
                                                                                                                                                        

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
  int iter;


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

  inline void update_D( int m, int l, const double *q_l, double alpha )
  {

    minusScaledFrom( D + m * LabelSet::classes, q + l * LabelSet::classes, LabelSet::classes, alpha );
    addScaledTo( D + m * LabelSet::classes, q_l, LabelSet::classes, alpha );
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
   *     sum_m ( D(m) - alpha(l,m) * q(l) + alpha(l,m) * q'(l) )^2
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
    double t2[LabelSet::classes];        
    


    // t0 = sum_m alpha(l,m) * D(m) 
    for ( auto& ele : _to_m ) {
      int m = ele.first;
      double alpha = ele.second;
      addScaledTo( t0, D + m * LabelSet::classes, LabelSet::classes, alpha );
    }


    
    // t0 += sum_m wt(l,j) (q(l) - q(j) )
    for ( auto& ele : _to_j ) {
      int j = ele.first;
      double wt = static_cast<double>( ele.second );
      minus( q + l * LabelSet::classes, q + j * LabelSet::classes, t1, LabelSet::classes );
      addScaledTo( t0, t1, LabelSet::classes, wt );
    }

    // negate t0 to get negative gradient direction
    negate( t0, LabelSet::classes );
    
    // Line Search
    // double energy_old = restrict_energy( l ) * options.wolf;

    // bool updated = false;

    
    // normalize_vec( t0, t0, LabelSet::classes );
    // double energy_new = 0.0;
    // for ( int i=0; i<40; i++ ) {
    //   scale( t0, LabelSet::classes, options.shrinkRatio );
    //   add( t0, q + l * LabelSet::classes, t1, LabelSet::classes );
    //   // Simplex Projection
    //   watershed( t1, t2, LabelSet::classes );

    //   energy_new = restrict_energy( l, t2 );
    //   if ( energy_new < energy_old ) {
    //     updated = true;
    //     break;
    //   }
    // }


    
    // Line Search Parabola


    double t3[LabelSet::classes];        
    bool updated = false;
    double a = 1.0; // initial step size
    double dE2 = norm2( t0, LabelSet::classes );
    double E0 = restrict_energy( l );

    memcpy( t1, q + l * LabelSet::classes, sizeof(double) * LabelSet::classes );
    addScaledTo( t1, t0, LabelSet::classes, a );
    watershed( t1, t2, LabelSet::classes );
    double E_a = restrict_energy( l, t2 );

    // printf( "E_a = %.4lf\n", E_a );

    // printf( "E0 = %.4lf\n", E0 );

    if ( E_a >= E0 ) {
      // Shrinking Branch
      for ( int i=0; i<40; i++ ) {
        a = ( a * a ) * dE2 / ( 2 * ( E_a - (E0 - dE2 * a ) ) );

        // update E_a
        memcpy( t1, q + l * LabelSet::classes, sizeof(double) * LabelSet::classes );
        addScaledTo( t1, t0, LabelSet::classes, a );
        watershed( t1, t2, LabelSet::classes );
        E_a = restrict_energy( l, t2 );
        
        if ( E_a < E0 ) {
          updated = true;
        }
      }
    } else { 
      // Expanding Branch
      
      updated = true;
      
      double E_best = E_a;
      for ( int i=0; i<40; i++ ) {
        if ( E_a > E0 - dE2 * a * ( 1 - 0.5 / 2.0 ) ) {
          double b = ( a * a ) * dE2 / ( 2 * ( E_a - (E0 - dE2 * a ) ) );
          
          // update E_a
          memcpy( t1, q + l * LabelSet::classes, sizeof(double) * LabelSet::classes );
          addScaledTo( t1, t0, LabelSet::classes, b );
          watershed( t1, t3, LabelSet::classes );
          E_a = restrict_energy( l, t3 );
    
          if ( E_a < E_best ) {
            E_best = E_a;
            memcpy( t2, t3, sizeof(double) * LabelSet::classes );
          }
        } else {
          a = a * 2.0;
        }

        memcpy( t1, q + l * LabelSet::classes, sizeof(double) * LabelSet::classes );
        addScaledTo( t1, t0, LabelSet::classes, a );
        watershed( t1, t3, LabelSet::classes );
        E_a = restrict_energy( l, t3 );
        
        if ( E_a < E_best ) {
          E_best = E_a;
          memcpy( t2, t3, sizeof(double) * LabelSet::classes );
        } else {
          break;
        }

      }
    }
    
    // debugging:
    // char ch;
    // scanf( "%c", &ch );
    

    if ( updated ) {
      for ( auto& ele : _to_m ) {
        int m = ele.first;
        double alpha = ele.second;
        update_D( m, l, t2, alpha );
      }
      memcpy( q + l * LabelSet::classes, t2, sizeof(double) * LabelSet::classes );
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

    // Update D(m) = sum alpha(l,m)*q(l,m) - P(m)
    for ( int m=0; m<M; m++ ) {
      update_D(m);
    }

    
    for ( iter=0; iter<options.maxIter; iter++ ) {
      // Update q(l)'s
      for ( int l=0; l<L; l++ ) {
        update_q(l);
      }

      Info( "Iteration %d - Energy: %.5lf\n", iter, total_energy() );
      
    }

    DeleteToNullWithTestArray( D );
    


  }


  void solve1( int M1, int L1,
               const Bipartite *m_to_l1,
               const double *P1, double *q1 )
  {
    M = M1;
    L = L1;
    P = P1;
    q = q1;
    m_to_l = m_to_l1;

    
    double *qp = q;
    for ( int l=0; l<L; l++ ) {
      auto &_to_m = m_to_l->getFromSet( l );
      memset( qp, 0, sizeof(double) * LabelSet::classes );
      for ( auto& ele : _to_m ) {
        int m = ele.first;
        double alpha = ele.second;
        addScaledTo( qp, P + m * LabelSet::classes, LabelSet::classes, alpha );
      }

      double s = sum_vec( qp, LabelSet::classes );
      
      scale( qp, LabelSet::classes, 1.0 / s );
      qp += LabelSet::classes;
    }
    
  }


};


int main( int argc, char **argv )
{
  
  if ( argc < 2 ) {
    Error( "Missing configuration file in options. (verify.conf)" );
    exit( -1 );
  }

  srand( 345645631 );
  

  env.parse( argv[1] );
  env.Summary();

  LabelSet::initialize( env["color-map"] );


  auto img = std::move( cvFeat<HOG>::gen( env["src-img"] ) );
  auto lbl = std::move( cvFeat<BGR>::gen( env["lbl-img"] ) );

  /// 1. Create an random partition of the patches in this image
  std::vector<std::pair<int,int> > training;
  std::vector<std::pair<int,int> > testing;

  for ( int i=7; i<img.rows-7; i++ ) {
    for ( int j=7; j<img.cols-7; j++ ) {
      if ( 0 == ( rand() & 1 ) ) {
        training.push_back( std::make_pair( i, j ) );
      } else {
        testing.push_back( std::make_pair( i, j ) );
      }
    }
  }
  
  /// 2. train the forest
  std::vector<FeatImage<float>::PatchProxy> l;

  for ( auto& every : training ) {
    l.push_back( img.Spawn( every.first, every.second ) );
  }
  
  timer::tic();
  Forest<SimpleKernel<float> > forest( 10, l, 0.5 );
  Done( "Tree built within %.5lf sec.", timer::utoc() );

  /// 3. Learn the leaf weights
  forest.PrepareWeitghts();
  float feat[img.GetPatchDim()];
  for ( auto& ele : training ) {
    img.FetchPatch( ele.first, ele.second, feat );
    forest.learn( feat );
  }
  timer::tic();
  for ( auto& ele : testing ) {
    img.FetchPatch( ele.first, ele.second, feat );
    forest.learn( feat );
  }
  Done( "Weights learned within %.5lf sec.", timer::utoc() );

  /// 4. Train Label


  std::vector<pair<int,int> > labeled;
  for ( auto& ele : training ) {
    if ( static_cast<double>( rand() ) / RAND_MAX < env["training-label-ratio"].toDouble() ) {
      labeled.push_back( ele );
    }
  }

  int M = static_cast<int>( labeled.size() );
  Bipartite m_to_l( M, forest.centers() );
  double *P = new double[ M * LabelSet::classes ];
  double *pP = P;
  int m = 0;
  for ( auto& ele : labeled ) {
    img.FetchPatch( ele.first, ele.second, feat );
    auto res = std::move( forest.query_with_coef( feat ) );
    for ( auto& item : res ) {
      m_to_l.add( m, item.first, item.second );
    }

    const uchar* color = lbl(ele.first,ele.second);
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

  Solver solver;
  solver.options.beta = 0.05;
  solver.options.maxIter = 10;

  double *q = new double[ forest.centers() * LabelSet::classes ];
  double *qp = q;
  for ( int i=0; i<forest.centers() * LabelSet::classes; i++ ) *(qp++) = LabelSet::inv;

  Info( "Solving ..." );
  solver.solve( M, forest.centers(), &forest, &m_to_l, P, q );
  Done( "Solved." );

  qp = q;
  for ( int l=0; l<forest.centers(); l++ ) {
    forest.updateLabelMap( l, qp );
    qp += LabelSet::classes;
  }

  DeleteToNullWithTestArray( q );
  DeleteToNullWithTestArray( P );
  Done( "Label Trained." );

  
  
  
  /// 5. Query
  
  GeoMap geomap = std::move( forest.PatchMatch( img ) );

  // 5.5 Verify PatchMatch is working
  // cv::Mat srcmat = cv::imread( env["src-img"] );
  // ImageViewer srcv( "source", srcmat );
  // ImageViewer refv( "reference", srcmat );

  // srcv.setCallback( [&geomap,&refv,&feat,&img]( int x, int y )
  //                   {
  //                     std::vector<PatLoc> list(1);
                        
  //                     img.FetchPatch( y, x, feat );

  //                     double minDist = 1000.0;
  //                     float feat_c[img.GetPatchDim()];
  //                     for ( auto& ele : geomap(y,x) ) {
  //                       PatLoc loc = ele.apply(y,x);
  //                       img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
  //                       double dist = dist_l2( feat, feat_c, img.GetPatchDim() );
  //                       if ( dist < minDist ) {
  //                         list[0] = loc;
  //                         minDist = dist;
  //                       }
  //                     }

                        
  //                     printf( "----------------------------------------\n" );
  //                     for ( auto& ele : list ) {
  //                       ele.show();
  //                     }
  //                     refv.display( list );
  //                   } );
    
  // while ( 27 != cv::waitKey(30) );


  // 5.6 Check Leaf Node
  // cv::Mat srcmat = cv::imread( env["src-img"] );
  // ImageViewer srcv( "source", srcmat );
  // ImageViewer refv( "reference", srcmat );
  // IconList iconlist( "leaf", 13 );
  // srcv.setCallback( [&geomap,&refv,&forest,&feat,&img,&iconlist,&srcmat]( int x, int y )
  //                   {
  //                     std::vector<PatLoc> list(1);
                        
  //                     img.FetchPatch( y, x, feat );

  //                     double minDist = 1000.0;
  //                     float feat_c[img.GetPatchDim()];
  //                     for ( auto& ele : geomap(y,x) ) {
  //                       PatLoc loc = ele.apply(y,x);
  //                       img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
  //                       double dist = dist_l2( feat, feat_c, img.GetPatchDim() );
  //                       if ( dist < minDist ) {
  //                         list[0] = loc;
  //                         minDist = dist;
  //                       }
  //                     }


  //                     auto res = std::move( forest.query( feat_c ) );
  //                     for ( auto& ele : res ) {
  //                       iconlist.clear();
  //                       for ( auto& item : forest(ele).store ) {
  //                         iconlist.push( srcmat, PatLoc( 0, item.y, item.x, 0.0f, 1.0f ) );
  //                       }
  //                       iconlist.display();
  //                       printVec( &forest(ele).q[0], LabelSet::classes );
  //                       while ( 32 != cv::waitKey(30) );
  //                     }

  //                     refv.display( list );
  //                   } );
    
  // while ( 27 != cv::waitKey(30) );
  



  /// 6. Vote
  
  cv::Mat estimated( geomap.rows, geomap.cols, CV_8UC3 );
  float feat_c[img.GetPatchDim()];  
  float vote[LabelSet::classes];
  for ( int i=0; i<geomap.rows; i++ ) {
    for ( int j=0; j<geomap.cols; j++ ) {
      double minDist = 1000.0;
      PatLoc best;
      img.FetchPatch( i, j, feat );
      
      for ( auto& ele : geomap(i,j) ) {
        PatLoc loc = ele.apply(i,j);
        img.FetchPatch( loc.y, loc.x, loc.rotation, loc.scale, feat_c );
        double dist = dist_l2( feat, feat_c, img.GetPatchDim() );
        if ( dist < minDist ) {
          minDist = dist;
          best = loc;
        }
      }
      img.FetchPatch( best.y, best.x, best.rotation, best.scale, feat_c );
      auto res = std::move( forest.query( feat_c ) );

      memset( vote, 0, sizeof(float) * LabelSet::classes );
      for ( auto& ele : res ) {
        addto( vote, &forest(ele).q[0], LabelSet::classes );
      }

      float highest = 0.0f;
      int classID = -1;
      for ( int k=0; k<LabelSet::classes; k++ ) {
        if ( vote[k] > highest ) {
          highest = vote[k];
          classID = k;
        }
      }
      
      
      auto& color = LabelSet::GetColor( classID );
      estimated.at<cv::Vec3b>(i,j)[0] = std::get<2>( color );
      estimated.at<cv::Vec3b>(i,j)[1] = std::get<1>( color );
      estimated.at<cv::Vec3b>(i,j)[2] = std::get<0>( color );
      
      
    }
  }

  cv::imshow( "result map", estimated );

  cv::imwrite( "result.png", estimated );
  
  while ( 27 != cv::waitKey(30) );
  
  return 0;
}


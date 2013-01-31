#pragma once

#include "../../data/Label.hpp"

namespace PatTk
{
  template <typename T>
  class EntropyKernel
  {
  public:
    typedef T dataType;

    class State
    {
    public:
      int *idx;
      int len;
      Shuffler shuffler;
      int depth;

      State( int *i, int l, int s, int d )
        : idx(i), len(l), shuffler(s), depth(d) {}
      
      State( int *i, int l, const Shuffler& s, int d )
        : idx(i), len(l), shuffler(s), depth(d) {}
    };


    class Judger
    {
    public:
      dataType th;
      int component;

      inline void write( FILE *out )
      {
        fwrite( &th, sizeof(dataType), 1, out );
        fwrite( &component, sizeof(int), 1, out );
      }

      inline void read( FILE *in )
      {
        fread( &th, sizeof(dataType), 1, in );
        fread( &component, sizeof(int), 1, in );
      }

      inline int operator()( const typename FeatImage<T>::PatchProxy &p ) const
      {
        if ( p(component) < th ) return 0;
        return 1;
      }

      inline int operator()( const T* p ) const
      {
        if ( p[component] < th ) return 0;
        return 1;
      }
    };

    // Kernel Parameters
    static int numHypo;
    static int stopNum;
    static typename Generalized<dataType>::type converge;


    static int split( const std::vector<typename FeatImage<T>::PatchProxy> &list,
                      const std::vector<typename FeatImage<float>::PatchProxy> &labels,
                      State& state,
                      Judger &judger,
                      int max_depth = -1 )
    {
      /* exception code:
       * -1 = too few patches within a node
       * -2 = no candidate component/dimension
       * -3 = all candidate dimension converges
       * -4, -5 = invalid split (totally unbalanced split)
       * -6 = max depth reached
       */
      if ( state.len <= stopNum ) {
        return -1;
      }
      if ( 0 == state.shuffler.Number() ) {
        return -2;
      }

      if ( max_depth == state.depth ) {
        return -6;
      }

      state.shuffler.ResetShuffle();
      
      int trial = 0;
      uint c[numHypo];
      dataType th[numHypo];
      while ( SHUFFLER_ERROR != ( c[trial] = state.shuffler.Next() ) && trial < numHypo ) {
        typename Generalized<T>::type min = list[state.idx[0]](c[trial]);
        typename Generalized<T>::type max = list[state.idx[0]](c[trial]);
        for ( int i=1; i<state.len; i++ ) {
          if ( list[state.idx[i]](c[trial]) > max ) {
            max = list[state.idx[i]](c[trial]);
          } else if ( list [state.idx[i]](c[trial]) < min ) {
            min = list[state.idx[i]](c[trial]);
          }
        }

        if ( max - min < converge ) {
          state.shuffler.Disqualify();
        } else {
          dataType range = max - min;
          th[trial] = rand() / static_cast<dataType>( RAND_MAX ) * range * 0.95 + range * 0.025 + min;
          trial++;
        }
      }
      
      if ( 0 == trial ) {
        return -3;
      }
      
      float lDistr[LabelSet::classes];
      float rDistr[LabelSet::classes];
      double minEnt = -1.0;
      int leftNumOpt = 0;
      int rightNumOpt = 0;
      for ( int t=0; t<trial; t++ ) {
        int leftNum = 0;
        int rightNum = 0;
        memset( lDistr, 0, sizeof(float) * LabelSet::classes );
        memset( rDistr, 0, sizeof(float) * LabelSet::classes );
        for ( int i=0; i<state.len; i++ ) {
          if ( list[state.idx[i]](c[t]) < th[t] ) {
            leftNum++;
            for ( int j=0; j<LabelSet::classes; j++ ) {
              lDistr[j] += labels[state.idx[i]](j);
            }
          } else {
            rightNum++;
            for ( int j=0; j<LabelSet::classes; j++ ) {
              rDistr[j] += labels[state.idx[i]](j);
            }
          }
        }

        double ent = leftNum * entropy( lDistr, LabelSet::classes );
        ent += rightNum * entropy( rDistr, LabelSet::classes );
        ent /= state.len;
        if ( minEnt < 0.0 || ent < minEnt ) {
          leftNumOpt = leftNum;
          rightNumOpt = rightNum;
          minEnt = ent;
          judger.th = th[t];
          judger.component = c[t];
        }
      }


      if ( 0 == leftNumOpt || 0 == rightNumOpt ) {
        return -4;
      }
      
      int right = -1;
      for ( int i=0; i<state.len; i++ ) {
        if ( 0 == judger( list[state.idx[i]] ) ) {
          right++;
          int tmp = state.idx[right];
          state.idx[right] = state.idx[i];
          state.idx[i] = tmp;
        }
      }

      return right + 1;
      
    }
  };

  template <typename T>
  int EntropyKernel<T>::numHypo = 10;

  template <typename T>
  int EntropyKernel<T>::stopNum = 1;

  template <>
  double EntropyKernel<float>::converge = 0.001;

  template <>
  double EntropyKernel<double>::converge = 0.001;

  template <>
  int EntropyKernel<unsigned char>::converge = 10;

}

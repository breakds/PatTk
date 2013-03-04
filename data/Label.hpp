/*********************************************************************************
 * File: Label.hpp
 * Description: Define the I/O and representation of Labels, Label Probability Map
 *              and related operations
 * by BreakDS, University of Wisconsin Madison, Tue Dec 11 10:55:44 CST 2012
 *********************************************************************************/

#pragma once

#include <cassert>
#include <vector>
#include <string>
#include <tuple>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/LispFormParser.hpp"

using std::vector;
using std::string;
using std::tuple;
using std::make_tuple;
using std::get;

namespace PatTk
{

  namespace LabelSet
  {
    int classes = -1; // uninitialized
    double inv = 1.0;
    namespace {
      // properties of LabelSet
      vector<tuple<unsigned char, unsigned char, unsigned char> > colors;
      vector<string> className;
    };

    void initialize( string filename )
    {
      LispFormParser lisp;
      lisp.parse( filename );
      className.clear();
      colors.clear();
      for ( auto& every : lisp ) {
        className.push_back( every );
        colors.push_back( make_tuple( static_cast<unsigned char>( lisp[every][0].toInt() ),
                                      static_cast<unsigned char>( lisp[every][1].toInt() ),
                                      static_cast<unsigned char>( lisp[every][2].toInt() ) ) );
      }
      className.push_back( "Void" );
      colors.push_back( make_tuple( 0, 0, 0 ) );
      classes = static_cast<int>( className.size() );
      inv = 1.0 / classes;
    }

    void Summary()
    {
      if ( -1 == classes ) {
        Error( "LabelSet has not been initialized yet." );
        exit( -1 );
      }

      Info( "Label Set Summary ..." );
      for ( int i=0; i<classes; i++ ) {
        printf( "%3d)%20s\t\t%hhu\t%hhu\t%hhu\n", i,
                className[i].c_str(),
                get<0>( colors[i] ),
                get<1>( colors[i] ),
                get<2>( colors[i] ) );
      }
    }

    inline const tuple<unsigned char, unsigned char, unsigned char>& GetColor( int index )
    {
      assert( 0 <= index && index < classes );
      return colors[index];
    }

    inline int GetClass( unsigned char b, unsigned char g, unsigned char r )
    {
      for ( int i=0; i<classes; i++ ) {
        if ( r == get<0>( colors[i] ) &&
             g == get<1>( colors[i] ) &&
             b == get<2>( colors[i] ) ) {
          return i;
        }
      }
      return (classes - 1);
    }
    
    inline const string& GetClassName( int index )
    {
      return className[index];
    }

  };
}


/*********************************************************************************
 * File: LazyBuffer.hpp
 * Description: An Object Container whose items are stored externally and will 
 *              be loaded into memroy on demand
 * by BreakDS, @ University of Wisconsin-Madison, May 3 2012
 * ======================================================================
 * Add FreeList Support, by BreakDS, May 7 2012
 * Add Glue(), by BreakDS, May 16 2012
 *********************************************************************************/
#pragma once

#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <inttypes.h>
#include "LLPack/utils/extio.hpp"

#define LAZY_BUFFER_INPUT 1
#define LAZY_BUFFER_CREATE 2

using std::string;
using std::vector;
using std::map;





class FreeList {
private:
  typedef struct tagLinks {
    int prev, next;
    inline tagLinks() : prev(-1), next(-1) {}
  } Links;
  
  int n, head, tail;
  vector<Links> lst;
public:

  inline FreeList( const int num ) : n(num), head(-1), tail(-1)
  {
    lst.resize( num );
  }

  inline FreeList() : n(0), head(-1), tail(-1), lst() {}

  inline void init( const int num ) {
    n = num;
    head = tail = -1;
    lst.resize( num );
  }

  inline void promote( const int x )
  {
    if ( -1 == head ) {
      assert( -1 == tail );
      head = x;
      tail = x;
    } else if ( x == head ) {
      return ;
    } else if ( x == tail ) {
      assert( -1 == lst[x].next );
      assert( 0 <= lst[x].prev );
      assert( -1 == lst[head].prev );
      
      lst[lst[x].prev].next = -1;
      lst[x].prev = -1;
      lst[x].next = head;
      lst[head].prev = x;
      head = x;
    } else if ( -1 == lst[head].prev && -1 == lst[tail].next ) {
      lst[x].next = head;
      lst[head].prev = x;
      head = x;
    } else {
      assert( 0 <= lst[x].prev && 0 <= lst[x].next );

      lst[lst[x].prev].next = lst[x].next;
      lst[lst[x].next].prev = lst[x].prev;
      
      lst[x].prev = -1;
      lst[x].next = head;
      lst[head].prev = x;
      head = x;
    }
  }

  inline int discard()
  {
    if ( -1 == tail ) return -1;
    int i = tail;
    assert( -1 == lst[i].next );
    if ( -1 == lst[i].prev ) {
      assert( head == tail );
      lst[i].next = -1;
      tail = -1;
      head = -1;
    } else {
      lst[lst[i].prev].next = -1;
      tail = lst[i].prev;
      lst[i].prev = -1;
    }
    return i;
  }
};



template <typename ItemType> // ItemType should implement read() / write() / ItemType(FILE*)
class LazyBuffer {
private:
  // pointer from index to object position
  vector<int> forward_ptr;
  // pointer from object position to index
  vector<int> backward_ptr;
  // offset records in the file
  vector<long> offsets;
  // the hash table 
  map<uint64_t, int> hash;
  
  // the actual data
  vector<ItemType> items;

  // FreeList
  FreeList fl;

  
  

  FILE *fdata;   // File Descriptor of the data
  int limit;     // Maximum number of loaded items, -1 == no limit


  int type; // LAZY_BUFFER_INPUT | LAZY_BUFFER_CREATE
public:
  // Filename for the data
  string offsetfile;
  string datafile;


  
  

  // The Constructor
  inline LazyBuffer( const string& offsetFile, const string& dataFile, int direction=LAZY_BUFFER_INPUT ) : limit(-1)
  {
    type = direction;
    datafile = dataFile;
    offsetfile = offsetFile;
    if ( LAZY_BUFFER_INPUT == direction ) {
      WITH_OPEN( in, offsetFile.c_str(), "rb" );
      int len = 0;
      fread( &len, sizeof(int), 1, in );
      forward_ptr.resize( len );
      offsets.resize( len );
      backward_ptr.clear();
      items.clear();
      hash.clear();
      for ( auto& item : forward_ptr ) item = -1; // the forward_ptr is zero
      for ( auto& item : offsets ) fread( &item, sizeof(long), 1, in );
      fdata = fopen( dataFile.c_str(), "rb" );
      // Initialize the FreeList
      fl.init( len );
      END_WITH( in );
    } else if ( LAZY_BUFFER_CREATE == direction ) {
      fl.init(0);
      forward_ptr.clear();
      backward_ptr.clear();
      offsets.clear();
      items.clear();
      hash.clear();
      fdata = nullptr;
    } else {
      Error( "LazyBuffer(): Wrong Parameter for direction in constructor." );
    }
  }

  inline ~LazyBuffer()
  {
    if ( nullptr != fdata ) fclose( fdata );
  }

  inline void SetLimit( const int l )
  {
    limit = l;
  }

  inline int push( const ItemType& item, const uint64_t key ) 
  {
    assert( LAZY_BUFFER_CREATE == type );
    auto iter = hash.find( key );
    if ( hash.end() == iter ) {
      int forward_ind = static_cast<int>( forward_ptr.size() );
      int backward_ind = static_cast<int>( backward_ptr.size() );

      forward_ptr.push_back( backward_ind );
      backward_ptr.push_back( forward_ind );
      items.push_back( item );
      hash[key] = forward_ind;
      assert( backward_ptr.size() == items.size() );
      return forward_ind;
    }
    return iter->second;
  }

  inline int push( const ItemType& item )
  {
    assert( LAZY_BUFFER_CREATE == type );
    // Calculate offset
    int forward_ind = static_cast<int>( forward_ptr.size() );
    int backward_ind = static_cast<int>( backward_ptr.size() );
    forward_ptr.push_back( backward_ind );
    backward_ptr.push_back( forward_ind );
    items.push_back( item );
    assert( backward_ptr.size() == items.size() );
    return forward_ind;
  }


  inline int size()
  {
    return static_cast<int>( forward_ptr.size() );
  }
  

  // Access and load on demand
  inline ItemType& operator[]( const int i )
  {
    assert( i < static_cast<int>( forward_ptr.size() ) );
    if ( -1 == forward_ptr[i] ) {
      assert( LAZY_BUFFER_INPUT == type );
      if ( -1 != limit && static_cast<int>( backward_ptr.size() ) >= limit ) {
        int pos = fl.discard();
        int unload = forward_ptr[pos];
        forward_ptr[pos] = -1;
        assert( -1 != unload );
        fseek( fdata, offsets[i], SEEK_SET );
        items[unload] = std::move( ItemType( fdata ) );
        forward_ptr[i] = unload;
        backward_ptr[unload] = i;
      } else {
        fseek( fdata, offsets[i], SEEK_SET );
        items.push_back( std::move( ItemType( fdata ) ) );
        forward_ptr[i] = static_cast<int>( items.size() - 1 );
        backward_ptr.push_back( i );
        assert( backward_ptr.size() == items.size() );
      }
      fl.promote( i );
    } else {
      assert( ( 0 <= forward_ptr[i] ) && ( forward_ptr[i] < static_cast<int>( items.size() ) ) );
    }
    return items[forward_ptr[i]];
  }


  // Const access and load on demand
  inline const ItemType& operator()( const int i )
  {
    assert( i < static_cast<int>( forward_ptr.size() ) );
    if ( -1 == forward_ptr[i] ) {
      assert( LAZY_BUFFER_INPUT == type );
      if ( -1 != limit && static_cast<int>( backward_ptr.size() ) >= limit ) {
        int pos = fl.discard();
        int unload = forward_ptr[pos];
        forward_ptr[pos] = -1;
        assert( -1 != unload );
        fseek( fdata, offsets[i], SEEK_SET );
        items[unload] = std::move( ItemType( fdata ) );
        forward_ptr[i] = unload;
        backward_ptr[unload] = i;
      } else {
        fseek( fdata, offsets[i], SEEK_SET );
        items.push_back( std::move( ItemType( fdata ) ) );
        forward_ptr[i] = static_cast<int>( items.size() - 1 );
        backward_ptr.push_back( i );
        assert( backward_ptr.size() == items.size() );
      }
      fl.promote( i );
    } else {
      assert( ( 0 <= forward_ptr[i] ) && ( forward_ptr[i] < static_cast<int>( items.size() ) ) );
    }
    return items[forward_ptr[i]];
  }

  inline void write() 
  {
    assert( LAZY_BUFFER_CREATE == type );

    
    // Write data
    WITH_OPEN( out, datafile.c_str(), "wb" );
    int len = static_cast<int>( forward_ptr.size() );
    fwrite( &len, sizeof(int), 1, out );
    for ( auto& ele : forward_ptr ) {
      offsets.push_back( ftell( out ) );
      items[ele].write( out );
    }
    END_WITH( out );

    // Write offsets file
    WITH_OPEN( out, offsetfile.c_str(), "wb" );
    assert( forward_ptr.size() == offsets.size() );
    int len = static_cast<int>( forward_ptr.size() );
    fwrite( &len, sizeof(int), 1, out );
    for ( auto& ele : offsets ) {
      fwrite( &ele, sizeof(long), 1, out );
    }
    END_WITH( out );
  }

  inline void evacuate()
  {
    assert( LAZY_BUFFER_INPUT == type );
    for ( auto& ele : backward_ptr ) {
      forward_ptr[ele] = -1;
      fl.discard();
    }
    backward_ptr.clear();
    items.clear();
  }

  inline void fullLoad() // automatically set limit to -1
  {
    assert( LAZY_BUFFER_INPUT == type );
    limit = -1;
    int len = 0;
    fread( &len, sizeof(int), 1, fdata );
    assert( len = static_cast<int>( forward_ptr.size() ) );
    for ( uint i=0; i<forward_ptr.size(); i++ ) {
      if ( -1 == forward_ptr[i] ) {
        forward_ptr[i] = static_cast<int>( backward_ptr.size() );
        backward_ptr.push_back( i );
        fseek( fdata, offsets[i], SEEK_SET );
        items.push_back( std::move( ItemType( fdata ) ) );
      }
    }
  }


  static void Glue( const std::string& offsetFile0, const std::string& dataFile0, 
                    const std::string& offsetFile1, const std::string& dataFile1 )
  {
    long shifted = 0;
    int len = 0;
    int len0 = 0;
    int len1 = 0;
    
    // Read Original Length
    WITH_OPEN( in0, dataFile0.c_str(), "rb" );
    fread( &len0, sizeof(int), 1, in0 );
    END_WITH( in0 );
    
    // Modify data file content
    WITH_OPEN( out0, dataFile0.c_str(), "ab" );
    WITH_OPEN( in1, dataFile1.c_str(), "rb" );
    shifted = ftell( out0 );
    fread( &len1, sizeof(int), 1, in1 );
    int c = 0;
    while ( EOF != ( c = fgetc( in1 ) ) ) fputc( c, out0 );
    len = len0 + len1;
    END_WITH( in1 );
    END_WITH( out0 );

    // Modify data file length
    WITH_OPEN( out0, dataFile0.c_str(), "r+b" );
    fseek( out0, 0, SEEK_SET );
    fwrite( &len, sizeof(int), 1, out0 );
    END_WITH( out0 );

    // Modify offset data
    WITH_OPEN( out0, offsetFile0.c_str(), "ab" );
    WITH_OPEN( in1, offsetFile1.c_str(), "rb" );
    fread( &len1, sizeof(int), 1, in1 );
    long tmp;
    for ( auto i=0; i<len1; i++ ) {
      fread( &tmp, sizeof(long), 1, in1 );
      tmp += shifted;
      tmp -= sizeof(int);
      fwrite( &tmp, sizeof(long), 1, out0 );
    }
    END_WITH( in1 );
    END_WITH( out0 );

    // Modify offset file length
    WITH_OPEN( out0, offsetFile0.c_str(), "r+b" );
    fseek( out0, 0, SEEK_SET );
    fwrite( &len, sizeof(int), 1, out0 );
    END_WITH( out0 );
  }
};




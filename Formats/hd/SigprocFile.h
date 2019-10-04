/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


#include <fstream>

#include "hd/DataSource.h"

class SigprocFile: public DataSource 
{
  public:

    SigprocFile (const char* filename, bool fswap);
    ~SigprocFile ();

    bool   get_error() const { return m_error != 0; }
    size_t get_data (size_t nsamps, char* data);

  private:

    inline unsigned char flip_4bit (unsigned char val)
    {
        return (((val & 0x3) << 6) |
                ((val & 0xC) << 2) |
                ((val & 0x30) >> 2) |
                ((val & 0xC0) >> 6));
    };

    inline unsigned char flip_2bit (unsigned char val)
    {
        return (((val & 0x0F) << 4) |
                ((val & 0xF0) >> 4));
    };

    std::ifstream m_file_stream;
    int           m_error;
    bool          first_time;
    float         offset;
    float         scale;
    bool          fswap;

};

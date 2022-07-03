/*  Written in 2016-2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

/*
 * Modified by Joris van Rantwijk to fit in a C++ class.
 */

#ifndef XOROSHIRO128PLUS_H_
#define XOROSHIRO128PLUS_H_

#include <stdint.h>

/* ==== Comments from the reference implementation: */

/* This is xoroshiro128+ 1.0, our best and fastest small-state generator
   for floating-point numbers, but its state space is large enough only
   for mild parallelism. We suggest to use its upper bits for
   floating-point generation, as it is slightly faster than
   xoroshiro128++/xoroshiro128**. It passes all tests we are aware of
   except for the four lower bits, which might fail linearity tests (and
   just those), so if low linear complexity is not considered an issue (as
   it is usually the case) it can be used to generate 64-bit outputs, too;
   moreover, this generator has a very mild Hamming-weight dependency
   making our test (http://prng.di.unimi.it/hwd.php) fail after 5 TB of
   output; we believe this slight bias cannot affect any application. If
   you are concerned, use xoroshiro128++, xoroshiro128** or xoshiro256+.

   We suggest to use a sign test to extract a random Boolean value, and
   right shifts to extract subsets of bits.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. 

   NOTE: the parameters (a=24, b=16, b=37) of this version give slightly
   better results in our test than the 2016 version (a=55, b=14, c=36).
*/

/* ==== Modified code: */

/**
 * Pseudo random number generator "xoroshiro128+"
 *
 * This code is based on the reference implementation,
 * modified by Joris van Rantwijk to fit in a C++ class.
 *
 * Source: http://prng.di.unimi.it/
 */
class Xoroshiro128plus
{

public:
    Xoroshiro128plus(uint64_t s0, uint64_t s1)
    {
        m_state[0] = s0;
        m_state[1] = s1;
    }

    inline uint64_t next()
    {
        const uint64_t s0 = m_state[0];
        uint64_t s1 = m_state[1];
        const uint64_t result = s0 + s1;

        s1 ^= s0;
        m_state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        m_state[1] = rotl(s1, 37);

        return result;
    }

private:
    static inline uint64_t rotl(uint64_t x, int k)
    {
        return (x << k) | (x >> (64 - k));
    }

    uint64_t m_state[2];
};

#endif  // XOROSHIRO128PLUS_H_

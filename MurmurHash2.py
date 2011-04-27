#!/usr/bin/env python

# Couldn't get pyhash or murmur packages built in Mac OSX 10.6
# so I wrote this shitty port of MurmurHash2.0, based on the 
# original implementation by Austin Appleby

class MurmurHash2:
    def __init__(self, seed=0x9747b28c):
        self.m = 0x5bd1e995
        self.r = 24
        self.seed = seed

    def hash32(self, input):
        data = bytearray(input)
        length = len(data)
        h = self.seed^length
        length4 = length/4
        
        for i in range(0, length4):
            i4 = i * 4
            k = (data[i4+0]&0xff) + ((data[i4+1]&0xff)<<8) +\
                    ((data[i4+2]&0xff)<<16) + ((data[i4+3]&0xff)<<24)
            k = k * self.m
            k = k^(k >> self.r)
            k = k * self.m
            h = h * self.m
            h = h^k

        if length % 4 == 1:
            h = h ^ (data[length &~ 3]&0xff)
        elif length % 4 == 2:
            h = h ^ (data[(length &~ 3) + 1]&0xff) << 8
        elif length % 4 == 3:
            h = h ^ (data[(length &~ 3) + 2]&0xff) << 16

        h = h ^ (h >> 13)
        h = h * self.m
        h = h ^ (h >> 15)

        return int(h % 2147483648)


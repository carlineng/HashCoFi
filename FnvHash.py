class fnv32:
    """ Fowler/Noll/Vo 32 bit hash. isthe.com/chongo/tech/comp/fnv
    Slightly modified.  Actually only a 31 bit hash, just mod the
    number by pow(2, 31) at the end"""
    def __init__(self):
        self.fnv_prime = 16777619
        self.fnv_offset_basis = 2166136261
        self.max_32bit_uint = 4294967296
        self.size = pow(2, 31)

    def fnv1(self, datastr):
        hashed = self.fnv_offset_basis
        for a in bytearray(datastr):
            hashed = (hashed * self.fnv_prime) % self.max_32bit_uint
            hashed = hashed ^ a
        
        hashed = hashed % self.size
        return hashed

    def fnv1a(self, datastr):
        hashed = self.fnv_offset_basis
        for a in bytearray(datastr):
            hashed = hashed ^ a
            hashed = (hashed * self.fnv_prime) % self.max_32bit_uint

        hashed = hashed % self.size
        return hashed


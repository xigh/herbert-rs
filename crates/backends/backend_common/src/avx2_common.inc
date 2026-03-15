// Common macros for AVX2 assembly kernels
// Symbol name mangling: Apple prefixes with underscore

#ifdef __APPLE__
#define SYMBOL(name) _##name
#else
#define SYMBOL(name) name
#endif

#ifndef MEMORY_UTILS
#define MEMORY_UTILS
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include <signal.h>

jmp_buf jump;

inline void segv (int sig)
{
  longjmp (jump, 1); 
}

inline int memcheck (void *x) 
{
  volatile char c;
  int illegal = 0;

  signal (SIGSEGV, segv);

  if (!setjmp (jump))
    c = *(char *) (x);
  else
    illegal = 1;

  signal (SIGSEGV, SIG_DFL);

  return (illegal);
}

#endif
#ifndef FELSENSTEIN_DEBUG_TOOLS_H
#define FELSENSTEIN_DEBUG_TOOLS_H

// DEBUG stuff
#ifdef DEBUG_PRINT
static inline void print_array_dbg(c_float_t* array, int N) {
  printf("%s", "DBG: ");
  for(int n = 0; n < N; n++) {
    printf("%.9g ", array[n]);
  }
  printf("\n");
}

static inline void print_array_dbg_loc(char* string, c_float_t* array, int N) {
  printf("%s (%s) ", "DBG:", string);
  for(int n = 0; n < N; n++) {
    printf("%.9g ", array[n]);
  }
  printf("\n");
}

static inline void print_array_dbg_loc_int8(char* string, int8_t* array, int N) {
  printf("%s (%s) ", "DBG:", string);
  for(int n = 0; n < N; n++) {
    printf("%i ", array[n]);
  }
  printf("\n");
}
#endif
#endif //FELSENSTEIN_DEBUG_TOOLS_H

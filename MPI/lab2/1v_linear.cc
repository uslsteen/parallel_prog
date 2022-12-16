#include "timer.hpp"
#include <cstdio>
#include <cstdlib>
#include <math.h>
const int DEFAULT_HEIGHT = 1000;
const int DEFAULT_WIDTH = 1000;

double **createGrid(int height, int width, double (*filler)(int, int));
void deleteGrid(double **grid, int height, int width);
void exportData(const char *filename, double **grid, int width, int height);
double defaultFiller(int line, int column);

int main(int argc, char **argv) {
  int height = DEFAULT_HEIGHT, width = DEFAULT_WIDTH;

  if (argc == 3) {
    height = atoi(argv[1]);
    width = atoi(argv[2]);
  }

  else
    printf(
        "Using default grid size %d×%d.\n"
        "If you want to specify size, use \x1b[1m%s [height] [width]\x1b[0m\n",
        DEFAULT_HEIGHT, DEFAULT_WIDTH, argv[0]);

  auto grid = createGrid(height, width, defaultFiller);

  timer::Timer time{};
  //long start_time = clk();

  for (int line = 2; line < height; line++) {
    for (int column = 0; column < width - 3; column++)
      grid[line][column] = sin(5 * grid[line - 2][column + 3]);
  }

  long duration = time.elapsed<timer::microsecs>();

  printf("Linear: executed in %ld ms for grid %d×%d\n", duration, height,
         width);

  exportData("result_linear.txt", grid, height, width);

  deleteGrid(grid, height, width);

  return 0;
}

double defaultFiller(int line, int column) { return 10 * line + column; }

double **createGrid(int height, int width, double (*filler)(int, int)) {
  double **grid = new double *[height];

  for (int line = 0; line < height; line++) {
    grid[line] = new double[width];

    for (int column = 0; column < height; column++)
      grid[line][column] = filler(line, column);
  }

  return grid;
}

void deleteGrid(double **grid, int height, int width) {
  for (int line = 0; line < height; line++)
    delete[] grid[line];

  delete[] grid;
}

void exportData(const char *filename, double **grid, int height, int width) {
  FILE *output_file = fopen(filename, "w");

  for (int line = 0; line < height; line++) {
    for (int column = 0; column < width; column++)
      fprintf(output_file, "%f ", grid[line][column]);

    fprintf(output_file, "\n");
  }

  fclose(output_file);
}
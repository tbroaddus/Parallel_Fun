/**
 * Parallel Merge Sort using MPI
 */

#include <iostream>
#include <memory>
#include <cmath>
#include <algorithm>

#define SIZE 100 

#include "mpi.h"


// Function declarations; defined below main()

void mergeSort(std::shared_ptr<int[]> local_arr, const int local_size, const int rank, const int world_size, const int max_height, std::shared_ptr<int[]>& sorted_arr);

void mergeSortHelper(std::shared_ptr<int[]> local_arr, const int local_size, const int rank, const int world_size, const int height, const int max_height, std::shared_ptr<int[]>& sorted_arr);

int* generateRandomArr(const int size);

void printArr(const int* arr, const int size);

int getParent(const int rank, const int height);



// main() driver

int main(int argc, char* argv[]) {

  srand(time(NULL));
  int rank, world_size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int local_size = static_cast<int>(std::ceil(SIZE / world_size));

  std::shared_ptr<int[]> unsorted_arr{nullptr};
  if (rank == 0) {
    unsorted_arr.reset(generateRandomArr(SIZE));
    std::cout << "Unsorted array:\n";
    printArr(unsorted_arr.get(), SIZE);
  }

  std::shared_ptr<int[]> local_arr{new int [local_size]};

  // Scatter 
  MPI_Scatter(unsorted_arr.get(), local_size, MPI_INT, local_arr.get(), local_size, MPI_INT, 0, MPI_COMM_WORLD); 

  // Sort
  std::shared_ptr<int[]> sorted_arr{nullptr};
  const int max_height{static_cast<int>(log2(world_size))-1};
  std::sort(local_arr.get(), local_arr.get()+local_size);
  mergeSort(local_arr, local_size, rank, world_size, max_height, sorted_arr);

  // Print sorted array
  if (rank == 0) {
    std::cout << "Sorted Array:\n";
    printArr(sorted_arr.get(), SIZE);
  }
  MPI_Finalize();
  return 0;
}


// Function definitions

void mergeSort(std::shared_ptr<int[]> local_arr, const int local_size, const int rank, const int world_size, const int max_height, std::shared_ptr<int[]>& sorted_arr) {
  // Calculate the parent
  const int height{0};
  const int parent{getParent(rank, height)};
  if (parent == rank) {
  // Current rank is the parent, recv and sort
    // Must check to see if it is receiving from a child.. end cases
    std::shared_ptr<int[]> new_arr{new int[local_size * 2]};
    MPI_Recv(new_arr.get(), local_size, MPI_INT, MPI_ANY_SOURCE, height, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::copy(local_arr.get(), local_arr.get() + local_size, new_arr.get() + local_size);
    std::sort(new_arr.get(), new_arr.get() + (local_size * 2));
    // If we are at the max height, move new_arr into sorted_arr and return
    if (height == max_height) {
      if (rank == 0) {
        sorted_arr = std::move(new_arr);
      }
      return;
    // Else, continue
    } else {
      mergeSortHelper(new_arr, local_size * 2, rank, world_size, height+1, max_height, sorted_arr);
    }
  } else {
    // Current rank is the child, send
    MPI_Send(local_arr.get(), local_size, MPI_INT, parent, height, MPI_COMM_WORLD);
    return;
  }
}


void mergeSortHelper(std::shared_ptr<int[]> local_arr, const int local_size, const int rank, const int world_size, const int height, const int max_height, std::shared_ptr<int[]>& sorted_arr) {
  // Calculate the parent
  const int parent{getParent(rank, height)};
  if (parent == rank) {
    // Current rank is the parent, recv and sort
    // TODO: Must check to see if it is receiving from a child.. end cases
    std::shared_ptr<int[]> new_arr{new int[local_size * 2]};
    MPI_Recv(new_arr.get(), local_size, MPI_INT, MPI_ANY_SOURCE, height, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::copy(local_arr.get(), local_arr.get() + local_size, new_arr.get() + local_size);
    std::sort(new_arr.get(), new_arr.get() + (local_size * 2));
    // If we are at the max height, move new_arr into sorted_arr and return
    if (height == max_height) {
      if (rank == 0) {
        sorted_arr = std::move(new_arr);
      }
      return;
    // Else, continue
    } else {
      mergeSortHelper(new_arr, local_size * 2, rank, world_size, height+1, max_height, sorted_arr);
    }
  } else {
    // Current rank is the child, send and return
    MPI_Send(local_arr.get(), local_size, MPI_INT, parent, height, MPI_COMM_WORLD);
    return;
  }
}


int* generateRandomArr(const int size) {
  int* int_arr = new int[size];
  for (int i = 0; i < size; i++) {
    int_arr[i] = rand() % 1000;
  }
  return int_arr;
}


void printArr(const int* arr, const int size) {
  for (int i = 0; i < size; i++) {
    std::cout << i << ": " << arr[i] << "\n";
  }
  std::cout << std::endl;
}


int getParent(const int rank, const int height) {
  // N spaces between parents. 
  // If the rank is divisable by this number, then the rank is a parent
  const int n_between_parents{static_cast<int>(pow(2, height+1))};
  if (rank % n_between_parents == 0) {
    return rank;
  } else {
    // Else, the parent will be the rank minus the space in between parents divided by two
    // Equivalent to rank - 2^height
    return (rank - (n_between_parents / 2));
  }
}
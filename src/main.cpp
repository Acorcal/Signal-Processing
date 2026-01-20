#include <iostream>
#include <armadillo>

int main() {
arma::mat A = arma::randu<arma::mat>(3,3);
arma::vec b = arma::randu<arma::vec>(3);
arma::vec x = arma::solve(A,b);

std::cout <<"A:\n" << A
          <<"b:\n" << b
          <<"x:\n" << x;
          

  return 0;
}

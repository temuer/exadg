/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2026 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// symmetric_eigenbasis_closed_form() against dealii::eigenvectors() (QL with implicit
// shifts): eigenvalues, spectral projectors sum_{a in group} N_a (x) N_a over groups of equal
// eigenvalues (independent of the basis chosen in a repeated eigenspace), orthonormality and
// the reconstruction A = sum_a lambda_a N_a (x) N_a. Matrices: C = F^T F, repeated and nearly
// repeated eigenvalues (I, rotated uniaxial stretch, splittings 1e-8 and 1e-13), an unsorted
// diagonal and a strongly graded matrix.

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <exadg/structure/material/library/symmetric_eigenbasis.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace ExaDG::Structure;

template<int dim>
dealii::Tensor<2, dim>
rotation(double const angle)
{
  dealii::Tensor<2, dim> R;
  if constexpr(dim == 2)
  {
    R[0][0] = R[1][1] = std::cos(angle);
    R[0][1]           = -std::sin(angle);
    R[1][0]           = std::sin(angle);
  }
  else
  {
    // rotation about the axis (1, 2, 3) / sqrt(14)
    dealii::Tensor<1, dim> const n = dealii::Tensor<1, dim>({1., 2., 3.}) / std::sqrt(14.);
    dealii::Tensor<2, dim>       K;
    K[0][1] = -n[2], K[0][2] = n[1], K[1][0] = n[2];
    K[1][2] = -n[0], K[2][0] = -n[1], K[2][1] = n[0];
    R = dealii::unit_symmetric_tensor<dim>() + std::sin(angle) * K + (1. - std::cos(angle)) * K * K;
  }
  return R;
}

template<int dim>
dealii::SymmetricTensor<2, dim>
rotated_diagonal(std::array<double, dim> const & d, double const angle)
{
  dealii::Tensor<2, dim> D;
  for(int i = 0; i < dim; ++i)
    D[i][i] = d[i];
  dealii::Tensor<2, dim> const R = rotation<dim>(angle);
  return dealii::symmetrize(R * D * dealii::transpose(R));
}

template<int dim>
bool
test()
{
  std::vector<std::pair<std::string, dealii::SymmetricTensor<2, dim>>> cases;
  cases.emplace_back("identity", dealii::unit_symmetric_tensor<dim>());
  {
    std::array<double, dim> d;
    d.fill(1.);
    d[0] = 2.25;
    cases.emplace_back("uniaxial stretch, rotated", rotated_diagonal<dim>(d, 0.7));
    for(double const split : {1e-8, 1e-13})
    {
      d.fill(3.);
      d[0] = 1.;
      d[1] = 1. + split;
      cases.emplace_back("splitting " + std::to_string(split), rotated_diagonal<dim>(d, 0.4));
    }
  }
  {
    dealii::SymmetricTensor<2, dim> D;
    for(int i = 0; i < dim; ++i)
      D[i][i] = (i == 0) ? 3. : (i == 1 ? 1. : 2.);
    cases.emplace_back("unsorted diagonal", D);
  }
  {
    dealii::SymmetricTensor<2, dim> G;
    for(int i = 0; i < dim; ++i)
      for(int j = 0; j <= i; ++j)
        G[i][j] = (i == j) ? std::pow(1e10, i) : 1e-3 * (i + j + 1);
    cases.emplace_back("graded", G);
  }
  for(unsigned int k = 0; k < 50; ++k)
  {
    dealii::Tensor<2, dim> F;
    for(int i = 0; i < dim; ++i)
      for(int j = 0; j < dim; ++j)
        F[i][j] = (i == j) + 0.4 * std::sin(1.3 * k + 2.1 * i + 0.7 * j + 0.3 * i * j);
    cases.emplace_back("C = F^T F, " + std::to_string(k),
                       dealii::symmetrize(dealii::transpose(F) * F));
  }

  bool passed = true;
  for(auto const & [name, A] : cases)
  {
    double const scale     = A.norm();
    auto const   eigen     = symmetric_eigenbasis_closed_form(A);
    auto const   reference = dealii::eigenvectors(A);

    double                          value_error = 0.;
    dealii::SymmetricTensor<2, dim> reconstruction;
    dealii::Tensor<2, dim>          NNt;
    for(int a = 0; a < dim; ++a)
    {
      value_error = std::max(value_error, std::abs(eigen[a].first - reference[a].first));
      reconstruction += eigen[a].first *
                        dealii::symmetrize(dealii::outer_product(eigen[a].second, eigen[a].second));
      for(int b = 0; b < dim; ++b)
        NNt[a][b] = eigen[a].second * eigen[b].second;
    }
    double const reconstruction_error = (reconstruction - A).norm();
    double const orthonormality_error =
      (NNt - dealii::Tensor<2, dim>(dealii::unit_symmetric_tensor<dim>())).norm();

    // groups of equal reference eigenvalues; the projector error is ~ eps scale / gap, so a
    // group closes at a relative gap of 1e-6
    double projector_error = 0.;
    int    first           = 0;
    while(first < dim)
    {
      int last = first + 1;
      while(last < dim and reference[last - 1].first - reference[last].first < 1e-6 * scale)
        ++last;
      dealii::Tensor<2, dim> P, P_ref;
      for(int a = first; a < last; ++a)
      {
        P += dealii::outer_product(eigen[a].second, eigen[a].second);
        P_ref += dealii::outer_product(reference[a].second, reference[a].second);
      }
      projector_error = std::max(projector_error, (P - P_ref).norm());
      first           = last;
    }

    bool const ok = value_error < 1e-14 * scale and reconstruction_error < 1e-14 * scale and
                    orthonormality_error < 1e-14 and projector_error < 1e-9;
    if(not ok)
    {
      std::cerr << "dim = " << dim << ", " << name << ": eigenvalue " << value_error / scale
                << ", reconstruction " << reconstruction_error / scale << ", orthonormality "
                << orthonormality_error << ", projector " << projector_error << std::endl;
      passed = false;
    }
  }
  return passed;
}

int
main()
{
  bool const passed = test<2>() and test<3>();

  if(passed)
    std::cout << "Symmetric eigenbasis checks: PASS" << std::endl;

  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

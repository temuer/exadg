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

#ifndef STRUCTURE_MATERIAL_LIBRARY_SYMMETRIC_EIGENBASIS
#define STRUCTURE_MATERIAL_LIBRARY_SYMMETRIC_EIGENBASIS

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace ExaDG
{
namespace Structure
{
/**
 * Eigensolver for the principal stretches of the Ogden model:
 *  - DealII: dealii::eigenvectors() with its default method (QL with implicit shifts)
 *  - ClosedForm: symmetric_eigenbasis_closed_form() below
 */
enum class EigenSolver
{
  DealII,
  ClosedForm
};

namespace internal
{
// t = tan(phi), c = cos(phi), s = sin(phi) of the Jacobi rotation zeroing a_pq of
// [[a_pp, a_pq], [a_pq, a_qq]] (smaller rotation angle, sqrt(theta^2 + 1) without overflow)
template<typename Number>
void
jacobi_rotation(Number const app,
                Number const apq,
                Number const aqq,
                Number &     t,
                Number &     c,
                Number &     s)
{
  Number const theta     = (aqq - app) / (Number(2.) * apq);
  Number const abs_theta = std::abs(theta);
  Number const root      = (abs_theta > Number(1.)) ?
                             abs_theta * std::sqrt(Number(1.) + Number(1.) / (theta * theta)) :
                             std::sqrt(theta * theta + Number(1.));
  t                      = (theta < Number(0.) ? Number(-1.) : Number(1.)) / (abs_theta + root);
  c                      = Number(1.) / std::sqrt(t * t + Number(1.));
  s                      = t * c;
}
} // namespace internal

/**
 * Eigenvalues and eigenvectors of a symmetric 2x2 or 3x3 tensor without iteration, with the
 * same return type and (descending) order as dealii::eigenvectors().
 *
 * 3D (cf. D. Eberly, A robust eigensolver for 3x3 symmetric matrices, 2014): with
 * B = A - q I, q = tr(A) / 3, p^2 = |B|^2 / 6, the eigenvalues of B / p are
 * 2 cos(acos(det(B / p) / 2) / 3 + 2 pi k / 3). Only the root farthest from the middle one
 * (the largest for det(B) >= 0, else the smallest) is taken from this formula: it stays well
 * conditioned also when the other two coincide. Its eigenvector w is the largest cross product
 * of two rows of B - beta I; the other two eigenpairs follow from one Jacobi rotation of B
 * restricted to the orthogonal complement of w, and all eigenvalues are Rayleigh quotients.
 * Repeated eigenvalues need no special case (any orthonormal basis of the eigenspace).
 *
 * 2D: one Jacobi rotation. A diagonal tensor is returned sorted, without any arithmetic.
 */
template<int dim, typename Number>
std::array<std::pair<Number, dealii::Tensor<1, dim, Number>>, dim>
symmetric_eigenbasis_closed_form(dealii::SymmetricTensor<2, dim, Number> const & A)
{
  static_assert(dim == 2 or dim == 3, "Only implemented for dim = 2 and dim = 3.");

  std::array<std::pair<Number, dealii::Tensor<1, dim, Number>>, dim> result{};

  Number a[dim][dim];
  Number scale = 0.;
  for(int i = 0; i < dim; ++i)
  {
    for(int j = 0; j < dim; ++j)
    {
      a[i][j] = A[i][j];
      scale   = std::max(scale, std::abs(a[i][j]));
    }
  }
  if(scale == Number(0.))
  {
    for(int i = 0; i < dim; ++i)
      result[i].second[i] = 1.;
    return result;
  }

  // vector[d][k] = d-th component of the k-th eigenvector
  Number value[dim];
  Number vector[dim][dim];
  Number off_diagonal = 0.;
  for(int i = 0; i < dim; ++i)
    for(int j = i + 1; j < dim; ++j)
      off_diagonal = std::max(off_diagonal, std::abs(a[i][j]));
  if(off_diagonal == Number(0.))
  {
    // diagonal, e.g. C at an axis-aligned stretch
    for(int i = 0; i < dim; ++i)
    {
      value[i] = a[i][i];
      for(int d = 0; d < dim; ++d)
        vector[d][i] = (d == i) ? 1. : 0.;
    }
  }
  else if constexpr(dim == 2)
  {
    Number t, c, s;
    internal::jacobi_rotation(a[0][0], a[0][1], a[1][1], t, c, s);
    value[0]     = a[0][0] - t * a[0][1];
    value[1]     = a[1][1] + t * a[0][1];
    vector[0][0] = c;
    vector[1][0] = -s;
    vector[0][1] = s;
    vector[1][1] = c;
  }
  else
  {
    // scaled to entries <= 1 and shifted
    Number const inv_scale = Number(1.) / scale;
    for(int i = 0; i < dim; ++i)
      for(int j = 0; j < dim; ++j)
        a[i][j] *= inv_scale;
    Number const q = (a[0][0] + a[1][1] + a[2][2]) / Number(3.);
    for(int i = 0; i < dim; ++i)
      a[i][i] -= q;
    Number const p_sq = (a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2] +
                         Number(2.) * (a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2])) /
                        Number(6.);
    if(p_sq == Number(0.))
    {
      for(int i = 0; i < dim; ++i)
      {
        result[i].first     = scale * q;
        result[i].second[i] = 1.;
      }
      return result;
    }

    Number w[3];
    {
      Number const p     = std::sqrt(p_sq);
      Number const inv_p = Number(1.) / p;
      Number const det   = inv_p * inv_p * inv_p *
                         (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[1][2]) -
                          a[0][1] * (a[0][1] * a[2][2] - a[1][2] * a[0][2]) +
                          a[0][2] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]));
      Number const half_det = std::min(Number(1.), std::max(Number(-1.), Number(0.5) * det));
      Number const phi      = std::acos(half_det) / Number(3.);
      Number const beta =
        Number(2.) * std::cos(half_det >= Number(0.) ? phi : phi + Number(2.0943951023931954923));
      Number const lambda = p * beta;

      // rows of B - lambda I, the largest of their cross products
      Number const r[3][3] = {{a[0][0] - lambda, a[0][1], a[0][2]},
                              {a[0][1], a[1][1] - lambda, a[1][2]},
                              {a[0][2], a[1][2], a[2][2] - lambda}};
      Number       norm_sq = -1.;
      for(int k = 0; k < 3; ++k)
      {
        int const    i    = (k == 2) ? 1 : 0;
        int const    j    = (k == 0) ? 1 : 2;
        Number const x[3] = {r[i][1] * r[j][2] - r[i][2] * r[j][1],
                             r[i][2] * r[j][0] - r[i][0] * r[j][2],
                             r[i][0] * r[j][1] - r[i][1] * r[j][0]};
        Number const n    = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        if(n > norm_sq)
        {
          norm_sq = n;
          for(int d = 0; d < 3; ++d)
            w[d] = x[d];
        }
      }
      // rows of rank <= 1: only for B = 0 up to roundoff
      if(not(norm_sq > Number(0.)))
        return dealii::eigenvectors(A);
      Number const inv_norm = Number(1.) / std::sqrt(norm_sq);
      for(int d = 0; d < 3; ++d)
        w[d] *= inv_norm;
    }

    // orthonormal basis U, V of the orthogonal complement of w
    Number U[3];
    if(std::abs(w[0]) > std::abs(w[1]))
    {
      Number const inv = Number(1.) / std::sqrt(w[0] * w[0] + w[2] * w[2]);
      U[0]             = -w[2] * inv;
      U[1]             = 0.;
      U[2]             = w[0] * inv;
    }
    else
    {
      Number const inv = Number(1.) / std::sqrt(w[1] * w[1] + w[2] * w[2]);
      U[0]             = 0.;
      U[1]             = w[2] * inv;
      U[2]             = -w[1] * inv;
    }
    Number const V[3] = {w[1] * U[2] - w[2] * U[1],
                         w[2] * U[0] - w[0] * U[2],
                         w[0] * U[1] - w[1] * U[0]};

    Number Bw[3], BU[3], BV[3];
    for(int i = 0; i < 3; ++i)
    {
      Bw[i] = a[i][0] * w[0] + a[i][1] * w[1] + a[i][2] * w[2];
      BU[i] = a[i][0] * U[0] + a[i][1] * U[1] + a[i][2] * U[2];
      BV[i] = a[i][0] * V[0] + a[i][1] * V[1] + a[i][2] * V[2];
    }
    Number const m_ww = w[0] * Bw[0] + w[1] * Bw[1] + w[2] * Bw[2];
    Number const m_uu = U[0] * BU[0] + U[1] * BU[1] + U[2] * BU[2];
    Number const m_uv = U[0] * BV[0] + U[1] * BV[1] + U[2] * BV[2];
    Number const m_vv = V[0] * BV[0] + V[1] * BV[1] + V[2] * BV[2];

    value[0] = m_ww;
    for(int d = 0; d < 3; ++d)
      vector[d][0] = w[d];
    if(m_uv == Number(0.))
    {
      value[1] = m_uu;
      value[2] = m_vv;
      for(int d = 0; d < 3; ++d)
      {
        vector[d][1] = U[d];
        vector[d][2] = V[d];
      }
    }
    else
    {
      Number t, c, s;
      internal::jacobi_rotation(m_uu, m_uv, m_vv, t, c, s);
      value[1] = m_uu - t * m_uv;
      value[2] = m_vv + t * m_uv;
      for(int d = 0; d < 3; ++d)
      {
        vector[d][1] = c * U[d] - s * V[d];
        vector[d][2] = s * U[d] + c * V[d];
      }
    }
    for(int k = 0; k < 3; ++k)
      value[k] = scale * (value[k] + q);
  }

  // sort descending
  for(int i = 0; i + 1 < dim; ++i)
  {
    for(int j = 0; j + 1 < dim - i; ++j)
    {
      if(value[j] < value[j + 1])
      {
        std::swap(value[j], value[j + 1]);
        for(int d = 0; d < dim; ++d)
          std::swap(vector[d][j], vector[d][j + 1]);
      }
    }
  }

  for(int k = 0; k < dim; ++k)
  {
    result[k].first = value[k];
    for(int d = 0; d < dim; ++d)
      result[k].second[d] = vector[d][k];
  }
  return result;
}

} // namespace Structure
} // namespace ExaDG

#endif /* STRUCTURE_MATERIAL_LIBRARY_SYMMETRIC_EIGENBASIS */

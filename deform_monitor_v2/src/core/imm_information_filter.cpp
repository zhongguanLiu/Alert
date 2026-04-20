/*
 * Author: zgliu@cumt.edu.cn
 * Affiliation: China University of Mining and Technology
 * Open-source release date: 2026-04-20
 */

#include "deform_monitor_v2/core/imm_information_filter.hpp"

#include "deform_monitor_v2/core/covariance_extractor.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace deform_monitor_v2 {

namespace {


double InverseNormalCdf_local(double p) {
  constexpr double a1 = -3.969683028665376e+01, a2 = 2.209460984245205e+02,
                   a3 = -2.759285104469687e+02, a4 = 1.383577518672690e+02,
                   a5 = -3.066479806614716e+01, a6 = 2.506628277459239e+00;
  constexpr double b1 = -5.447609879822406e+01, b2 = 1.615858368580409e+02,
                   b3 = -1.556989798598866e+02, b4 = 6.680131188771972e+01,
                   b5 = -1.328068155288572e+01;
  constexpr double c1 = -7.784894002430293e-03, c2 = -3.223964580411365e-01,
                   c3 = -2.400758277161838e+00, c4 = -2.549732539343734e+00,
                   c5 = 4.374664141464968e+00, c6 = 2.938163982698783e+00;
  constexpr double d1 = 7.784695709041462e-03, d2 = 3.224671290700398e-01,
                   d3 = 2.445134137142996e+00, d4 = 3.754408661907416e+00;
  constexpr double plow = 0.02425, phigh = 1.0 - 0.02425;
  if (p <= 0.0) return -std::numeric_limits<double>::infinity();
  if (p >= 1.0) return std::numeric_limits<double>::infinity();
  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
           ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
  }
  if (p > phigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
           ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
  }
  const double q = p - 0.5, r = q * q;
  return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
         (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0);
}

double Chi2ThresholdByDof(int dof, double alpha_s) {
  if (dof <= 0) return std::numeric_limits<double>::infinity();
  const double p = std::max(1.0e-6, std::min(1.0 - 1.0e-6, 1.0 - alpha_s));
  const double z = InverseNormalCdf_local(p);
  const double k = static_cast<double>(dof);
  const double base = 1.0 - 2.0/(9.0*k) + z*std::sqrt(2.0/(9.0*k));
  return k * base * base * base;
}

Eigen::Matrix<double, 6, 6> SolveSPD(const Eigen::Matrix<double, 6, 6>& A) {
  Eigen::Matrix<double, 6, 6> S = 0.5 * (A + A.transpose());
  Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(S);
  if (ldlt.info() != Eigen::Success) {
    S += Eigen::Matrix<double, 6, 6>::Identity() * 1.0e-6;
    ldlt.compute(S);
  }
  return ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());
}

void DampenModelState(ModelState* model, double disp_decay, double vel_decay) {
  if (!model) {
    return;
  }
  model->x.block<3, 1>(0, 0) *= disp_decay;
  model->x.block<3, 1>(3, 0) *= vel_decay;
}

Eigen::Matrix3d TypeConstraintInfo(const AnchorReference& anchor,
                                   double lambda_suppressed,
                                   const Eigen::Vector3d& prior_u,
                                   Eigen::Vector3d* constrained_ref_u) {
  const Eigen::Vector3d e1 = anchor.basis_R.col(0).normalized();
  const Eigen::Vector3d e2 = anchor.basis_R.col(1).normalized();
  const Eigen::Vector3d n = anchor.basis_R.col(2).normalized();

  Eigen::Matrix3d projector_allowed = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d projector_suppressed = Eigen::Matrix3d::Zero();
  if (anchor.type == AnchorType::PLANE) {
    projector_allowed = n * n.transpose();
    projector_suppressed = e1 * e1.transpose() + e2 * e2.transpose();
  } else if (anchor.type == AnchorType::EDGE) {
    projector_allowed = e1 * e1.transpose() + n * n.transpose();
    projector_suppressed = e2 * e2.transpose();
  } else {
    projector_allowed = e2 * e2.transpose() + n * n.transpose();
    projector_suppressed = e1 * e1.transpose();
  }

  if (constrained_ref_u) {
    *constrained_ref_u = projector_allowed * prior_u;
  }
  return lambda_suppressed * projector_suppressed;
}

}  // namespace

void ImmInformationFilter::SetParams(const ImmParams& imm_params,
                                     const ObservabilityParams& observability_params,
                                     const SignificanceParams& significance_params,
                                     const DirectionalMotionParams& directional_params,
                                     double tau_mu0) {
  imm_params_ = imm_params;
  observability_params_ = observability_params;
  significance_params_ = significance_params;
  directional_params_ = directional_params;
  tau_mu0_ = tau_mu0;
}

void ImmInformationFilter::InitializeAnchorState(AnchorTrackState* state) const {
  if (!state) {
    return;
  }
  state->model0.x.setZero();
  state->model1.x.setZero();
  state->model0.P.setIdentity();
  state->model1.P.setIdentity();
  state->model0.P.block<3, 3>(0, 0) *= 0.05 * 0.05;
  state->model0.P.block<3, 3>(3, 3) *= 0.02 * 0.02;
  state->model1.P = state->model0.P;
  state->model0.mu = imm_params_.enable_model_competition ? tau_mu0_ : 0.0;
  state->model1.mu = imm_params_.enable_model_competition ? (1.0 - tau_mu0_) : 1.0;
  state->x_mix.setZero();
  state->P_mix = state->model1.P;
  state->chi2_stat = 0.0;
  state->disp_norm = 0.0;
  state->disp_normal = 0.0;
  state->disp_edge = 0.0;
  state->cusum_score = 0.0;
  state->disappearance_score = 0.0;
  state->dof_obs = 0;
  state->comparable = false;
  state->observable = false;
  state->gate_state = ObsGateState::NOT_OBSERVABLE;
  state->significant = false;
  state->persistent_candidate = false;
  state->disappearance_candidate = false;
  state->reacquired = false;
  state->mode = DetectionMode::NONE;
  state->stable_streak = 0;
  state->disappearance_streak = 0;
  state->dead_count = 0;
  state->cluster_member = false;
  state->matched_center_R.setZero();
  state->cusum_history.clear();
  state->evidence_history.clear();

  state->directional_S.setZero();
  state->directional_quality_sum = 0.0;
  state->directional_persistent = false;
  state->D_max.setZero();
  state->permanent_deformed = false;
}

void ImmInformationFilter::MixModelStates(const AnchorTrackState& state,
                                          Eigen::Matrix<double, 6, 1>* x0,
                                          Eigen::Matrix<double, 6, 6>* P0,
                                          Eigen::Matrix<double, 6, 1>* x1,
                                          Eigen::Matrix<double, 6, 6>* P1,
                                          double* mu0_pred,
                                          double* mu1_pred) const {
  const double mu0 = state.model0.mu;
  const double mu1 = state.model1.mu;
  *mu0_pred = imm_params_.p00 * mu0 + imm_params_.p10 * mu1;
  *mu1_pred = imm_params_.p01 * mu0 + imm_params_.p11 * mu1;
  *mu0_pred = std::max(*mu0_pred, 1.0e-9);
  *mu1_pred = std::max(*mu1_pred, 1.0e-9);

  const double mu00 = imm_params_.p00 * mu0 / *mu0_pred;
  const double mu10 = imm_params_.p10 * mu1 / *mu0_pred;
  const double mu01 = imm_params_.p01 * mu0 / *mu1_pred;
  const double mu11 = imm_params_.p11 * mu1 / *mu1_pred;

  *x0 = mu00 * state.model0.x + mu10 * state.model1.x;
  *x1 = mu01 * state.model0.x + mu11 * state.model1.x;

  *P0 = mu00 * (state.model0.P + (state.model0.x - *x0) * (state.model0.x - *x0).transpose()) +
        mu10 * (state.model1.P + (state.model1.x - *x0) * (state.model1.x - *x0).transpose());
  *P1 = mu01 * (state.model0.P + (state.model0.x - *x1) * (state.model0.x - *x1).transpose()) +
        mu11 * (state.model1.P + (state.model1.x - *x1) * (state.model1.x - *x1).transpose());
}

void ImmInformationFilter::Predict(AnchorTrackState* state, double dt) const {
  if (!state) {
    return;
  }
  const double clamped_dt = std::max(1.0e-3, std::min(0.5, dt));

  Eigen::Matrix<double, 6, 6> F1 = Eigen::Matrix<double, 6, 6>::Identity();
  F1.block<3, 3>(0, 3) = clamped_dt * Eigen::Matrix3d::Identity();

  Eigen::Matrix<double, 6, 6> Q1 = Eigen::Matrix<double, 6, 6>::Zero();
  Q1.block<3, 3>(0, 0) = imm_params_.q_u1 * clamped_dt * Eigen::Matrix3d::Identity();
  Q1.block<3, 3>(3, 3) = imm_params_.q_v1 * clamped_dt * Eigen::Matrix3d::Identity();

  if (!imm_params_.enable_model_competition) {
    state->model1.x = F1 * state->model1.x;
    state->model1.P = 0.5 * ((F1 * state->model1.P * F1.transpose() + Q1) +
                             (F1 * state->model1.P * F1.transpose() + Q1).transpose());
    state->model0.x = state->model1.x;
    state->model0.P = state->model1.P;
    state->model0.mu = 0.0;
    state->model1.mu = 1.0;
    state->x_mix = state->model1.x;
    state->P_mix = 0.5 * (state->model1.P + state->model1.P.transpose());
    return;
  }

  Eigen::Matrix<double, 6, 1> x0_mix;
  Eigen::Matrix<double, 6, 1> x1_mix;
  Eigen::Matrix<double, 6, 6> P0_mix;
  Eigen::Matrix<double, 6, 6> P1_mix;
  double mu0_pred = 0.5;
  double mu1_pred = 0.5;
  MixModelStates(*state, &x0_mix, &P0_mix, &x1_mix, &P1_mix, &mu0_pred, &mu1_pred);

  Eigen::Matrix<double, 6, 6> F0 = Eigen::Matrix<double, 6, 6>::Identity();
  F0.block<3, 3>(0, 3) = clamped_dt * Eigen::Matrix3d::Identity();
  F0.block<3, 3>(3, 3) = imm_params_.rho * Eigen::Matrix3d::Identity();

  Eigen::Matrix<double, 6, 6> Q0 = Eigen::Matrix<double, 6, 6>::Zero();
  Q0.block<3, 3>(0, 0) = imm_params_.q_u0 * clamped_dt * Eigen::Matrix3d::Identity();
  Q0.block<3, 3>(3, 3) = imm_params_.q_v0 * clamped_dt * Eigen::Matrix3d::Identity();

  state->model0.x = F0 * x0_mix;
  state->model1.x = F1 * x1_mix;
  state->model0.P = 0.5 * ((F0 * P0_mix * F0.transpose() + Q0) +
                           (F0 * P0_mix * F0.transpose() + Q0).transpose());
  state->model1.P = 0.5 * ((F1 * P1_mix * F1.transpose() + Q1) +
                           (F1 * P1_mix * F1.transpose() + Q1).transpose());
  state->model0.mu = mu0_pred;
  state->model1.mu = mu1_pred;

  const double norm_mu = state->model0.mu + state->model1.mu;
  state->model0.mu /= norm_mu;
  state->model1.mu /= norm_mu;

  state->x_mix = state->model0.mu * state->model0.x + state->model1.mu * state->model1.x;
  state->P_mix =
      state->model0.mu * (state->model0.P +
                          (state->model0.x - state->x_mix) *
                              (state->model0.x - state->x_mix).transpose()) +
      state->model1.mu * (state->model1.P +
                          (state->model1.x - state->x_mix) *
                              (state->model1.x - state->x_mix).transpose());
  state->P_mix = 0.5 * (state->P_mix + state->P_mix.transpose());
}

void ImmInformationFilter::Update(AnchorTrackState* state,
                                  const AnchorReference& anchor,
                                  const CurrentObservation& obs) const {
  if (!state) {
    return;
  }

  state->comparable = obs.comparable;
  state->observable = obs.observable;
  state->gate_state = obs.gate_state;
  if (!state->comparable || obs.scalars.empty()) {
    ++state->dead_count;
    if (imm_params_.enable_model_competition) {
      const double disp_decay = state->dead_count > 3 ? 0.85 : 0.95;
      const double vel_decay = state->dead_count > 1 ? 0.20 : 0.50;
      DampenModelState(&state->model0, disp_decay, vel_decay);
      DampenModelState(&state->model1, disp_decay, vel_decay);
      state->model0.mu = std::max(state->model0.mu, tau_mu0_);
      state->model1.mu = std::min(state->model1.mu, 1.0 - tau_mu0_);
      const double mu_sum = std::max(1.0e-9, state->model0.mu + state->model1.mu);
      state->model0.mu /= mu_sum;
      state->model1.mu /= mu_sum;
      state->x_mix = state->model0.mu * state->model0.x + state->model1.mu * state->model1.x;
      state->P_mix =
          state->model0.mu * (state->model0.P +
                              (state->model0.x - state->x_mix) *
                                  (state->model0.x - state->x_mix).transpose()) +
          state->model1.mu * (state->model1.P +
                              (state->model1.x - state->x_mix) *
                                  (state->model1.x - state->x_mix).transpose());
      state->P_mix = 0.5 * (state->P_mix + state->P_mix.transpose());
    } else {
      state->model0.x = state->model1.x;
      state->model0.P = state->model1.P;
      state->model0.mu = 0.0;
      state->model1.mu = 1.0;
      state->x_mix = state->model1.x;
      state->P_mix = 0.5 * (state->model1.P + state->model1.P.transpose());
    }
    state->dof_obs = 0;
    state->chi2_stat = Chi2PseudoInverse(state->x_mix.block<3, 1>(0, 0),
                                         state->P_mix.block<3, 3>(0, 0));
    return;
  }
  state->dead_count = 0;

  const int M = static_cast<int>(obs.scalars.size());
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(M, 3);
  Eigen::VectorXd z = Eigen::VectorXd::Zero(M);
  Eigen::VectorXd r_diag = Eigen::VectorXd::Zero(M);
  for (int i = 0; i < M; ++i) {
    H.row(i) = obs.scalars[i].h_R.transpose();
    z(i) = obs.scalars[i].z;
    r_diag(i) = std::max(1.0e-9, obs.scalars[i].r);
  }

  Eigen::Matrix3d Lambda_meas = Eigen::Matrix3d::Zero();
  for (int i = 0; i < M; ++i) {
    Lambda_meas += (obs.scalars[i].h_R * obs.scalars[i].h_R.transpose()) /
                   std::max(1.0e-9, obs.scalars[i].r);
  }
  Lambda_meas = 0.5 * (Lambda_meas + Lambda_meas.transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> meas_eig(Lambda_meas);
  state->dof_obs = 0;
  if (meas_eig.info() == Eigen::Success) {
    for (int i = 0; i < 3; ++i) {
      if (meas_eig.eigenvalues()(i) > observability_params_.tau_lambda) {
        ++state->dof_obs;
      }
    }
  }
  if (state->dof_obs == 0) {
    state->chi2_stat = Chi2PseudoInverse(state->x_mix.block<3, 1>(0, 0),
                                         state->P_mix.block<3, 3>(0, 0));
    return;
  }

  Eigen::MatrixXd Hbar = Eigen::MatrixXd::Zero(M, 6);
  Hbar.leftCols<3>() = H;

  auto update_model = [&](ModelState* model, double* log_likelihood) {
    const Eigen::Matrix<double, 6, 1> x_minus = model->x;
    const Eigen::Matrix<double, 6, 6> P_minus = model->P;
    const Eigen::Matrix<double, 6, 6> Lambda_minus = SolveSPD(P_minus);
    const Eigen::Matrix<double, 6, 1> eta_minus = Lambda_minus * x_minus;

    Eigen::Matrix<double, 6, 6> Lambda_obs = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> eta_obs = Eigen::Matrix<double, 6, 1>::Zero();
    for (int i = 0; i < M; ++i) {
      const double inv_r = 1.0 / r_diag(i);
      const Eigen::Matrix<double, 6, 1> h = Hbar.row(i).transpose();
      Lambda_obs += inv_r * (h * h.transpose());
      eta_obs += inv_r * h * z(i);
    }

    if (imm_params_.enable_type_constraint) {
      Eigen::Vector3d constrained_ref_u = Eigen::Vector3d::Zero();
      const double lambda_suppressed =
          anchor.type == AnchorType::PLANE ? 400.0 : 220.0;
      const Eigen::Matrix3d Lambda_type =
          TypeConstraintInfo(anchor,
                             lambda_suppressed,
                             x_minus.block<3, 1>(0, 0),
                             &constrained_ref_u);
      Lambda_obs.block<3, 3>(0, 0) += Lambda_type;
      eta_obs.block<3, 1>(0, 0) += Lambda_type * constrained_ref_u;
    }

    const Eigen::Matrix<double, 6, 6> Lambda_plus =
        0.5 * ((Lambda_minus + Lambda_obs) + (Lambda_minus + Lambda_obs).transpose());
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(Lambda_plus);
    Eigen::Matrix<double, 6, 6> Lambda_plus_reg = Lambda_plus;
    if (ldlt.info() != Eigen::Success) {
      Lambda_plus_reg += Eigen::Matrix<double, 6, 6>::Identity() * 1.0e-9;
      ldlt.compute(Lambda_plus_reg);
    }
    const Eigen::Matrix<double, 6, 1> eta_plus = eta_minus + eta_obs;
    const Eigen::Matrix<double, 6, 1> x_plus = ldlt.solve(eta_plus);
    const Eigen::Matrix<double, 6, 6> P_plus =
        ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());

    Eigen::MatrixXd R = r_diag.asDiagonal();
    Eigen::MatrixXd S = Hbar * P_minus * Hbar.transpose() + R;
    S = Symmetrize(S);
    S += Eigen::MatrixXd::Identity(M, M) * 1.0e-9;
    Eigen::LDLT<Eigen::MatrixXd> S_ldlt(S);
    if (S_ldlt.info() != Eigen::Success) {
      S += Eigen::MatrixXd::Identity(M, M) * 1.0e-6;
      S_ldlt.compute(S);
    }
    const Eigen::VectorXd nu = z - Hbar * x_minus;
    const double quad = nu.dot(S_ldlt.solve(nu));
    double log_det = 0.0;
    const auto D = S_ldlt.vectorD();
    for (int i = 0; i < D.size(); ++i) {
      log_det += std::log(std::max(1.0e-12, D(i)));
    }
    *log_likelihood = -0.5 * (quad + log_det + static_cast<double>(M) * std::log(2.0 * M_PI));

    model->x = x_plus;
    model->P = 0.5 * (P_plus + P_plus.transpose());
  };

  if (!imm_params_.enable_model_competition) {
    double logL1 = -1.0e9;
    update_model(&state->model1, &logL1);
    state->model0.x = state->model1.x;
    state->model0.P = state->model1.P;
    state->model0.mu = 0.0;
    state->model1.mu = 1.0;
    state->x_mix = state->model1.x;
    state->P_mix = 0.5 * (state->model1.P + state->model1.P.transpose());
    state->chi2_stat =
        Chi2PseudoInverse(state->x_mix.block<3, 1>(0, 0), state->P_mix.block<3, 3>(0, 0));
    return;
  }

  double logL0 = -1.0e9;
  double logL1 = -1.0e9;
  update_model(&state->model0, &logL0);
  update_model(&state->model1, &logL1);

  const double max_log = std::max(logL0, logL1);
  const double w0 = std::exp(logL0 - max_log) * state->model0.mu;
  const double w1 = std::exp(logL1 - max_log) * state->model1.mu;
  const double norm = std::max(1.0e-12, w0 + w1);
  state->model0.mu = w0 / norm;
  state->model1.mu = w1 / norm;

  state->x_mix = state->model0.mu * state->model0.x + state->model1.mu * state->model1.x;
  state->P_mix =
      state->model0.mu * (state->model0.P +
                          (state->model0.x - state->x_mix) *
                              (state->model0.x - state->x_mix).transpose()) +
      state->model1.mu * (state->model1.P +
                          (state->model1.x - state->x_mix) *
                              (state->model1.x - state->x_mix).transpose());
  state->P_mix = 0.5 * (state->P_mix + state->P_mix.transpose());
  state->chi2_stat =
      Chi2PseudoInverse(state->x_mix.block<3, 1>(0, 0), state->P_mix.block<3, 3>(0, 0));
}

void ImmInformationFilter::UpdateCusum(AnchorTrackState* state) const {
  if (!state) {
    return;
  }
  if (!significance_params_.enable_cusum) {
    state->cusum_score = 0.0;
    state->cusum_history.push_back(0.0);
    while (state->cusum_history.size() > 20) {
      state->cusum_history.pop_front();
    }
    state->persistent_candidate = false;
    return;
  }
  if (!state->comparable || state->dof_obs <= 0) {
    state->cusum_score *= 0.85;
    if (state->cusum_score < significance_params_.cusum_k) {
      state->cusum_score = 0.0;
    }
    state->cusum_history.push_back(state->cusum_score);
    while (state->cusum_history.size() > 20) {
      state->cusum_history.pop_front();
    }
    state->persistent_candidate = false;
    return;
  }







  const int dof_eff = std::max(1, state->dof_obs);
  const double chi2_threshold =
      Chi2ThresholdByDof(dof_eff, significance_params_.alpha_s);
  const double cusum_input =
      (state->chi2_stat > chi2_threshold) ? 1.0 : -0.5;
  const double lambda = significance_params_.cusum_lambda;
  const double cap = significance_params_.cusum_cap_factor *
                     significance_params_.cusum_h;
  state->cusum_score =
      std::min(cap,
               std::max(0.0, lambda * state->cusum_score + cusum_input));
  state->cusum_history.push_back(state->cusum_score);
  while (state->cusum_history.size() > 20) {
    state->cusum_history.pop_front();
  }
  state->persistent_candidate = state->cusum_score > significance_params_.cusum_h;
}

void ImmInformationFilter::UpdateDirectionalMotion(AnchorTrackState* state,
                                                   const AnchorReference& anchor,
                                                   double cmp_score,
                                                   double dt) const {
  if (!state) {
    return;
  }
  if (!directional_params_.enable) {
    state->directional_S.setZero();
    state->directional_quality_sum = 0.0;
    state->directional_persistent = false;
    return;
  }


  const Eigen::Vector3d u = state->x_mix.block<3, 1>(0, 0);
  const double u_norm = u.norm();
  const double q = std::max(0.0, std::min(1.0, state->comparable ? cmp_score : 0.0));


  const double clamped_dt = std::max(1.0e-3, std::min(5.0, dt));
  const double lambda_dt = std::pow(directional_params_.lambda0, clamped_dt);

  double w = q;
  if (state->directional_S.norm() >= directional_params_.epsilon && u_norm > 1.0e-9) {
    const double cos_theta = state->directional_S.normalized().dot(u.normalized());
    w = q * cos_theta;
  }

  state->directional_S = lambda_dt * state->directional_S + w * u;
  state->directional_quality_sum = lambda_dt * state->directional_quality_sum + q;


  const double s_norm = state->directional_S.norm();
  const double quality_sum = std::max(1.0e-9, state->directional_quality_sum);
  state->directional_persistent =
      s_norm >= directional_params_.tau_s &&
      (s_norm / quality_sum) >= directional_params_.tau_c;



  if (state->directional_persistent && u_norm > state->D_max.norm()) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(anchor.Sigma_ref_geom);
    double sigma_anchor = 0.003;
    if (eig.info() == Eigen::Success) {
      sigma_anchor = std::max(0.003, std::sqrt(std::max(0.0, eig.eigenvalues()(2))));
    }
    if (u_norm >= directional_params_.tau_d * sigma_anchor) {
      state->D_max = u;
      state->permanent_deformed = true;
    }
  }
}

}  // namespace deform_monitor_v2

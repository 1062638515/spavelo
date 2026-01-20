import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np
from kernel import CauchyKernel, SampleKernel, BatchedCauchyKernel, EQKernel


def _add_diagonal_jitter(matrix, jitter=1e-8):
    Eye = torch.eye(matrix.size(-1), device=matrix.device).expand(matrix.shape)
    return matrix + jitter * Eye


class SVGP(nn.Module):
    def __init__(self, fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, jitter, N_train, dtype, device, task="standard", group_number=1):
        super(SVGP, self).__init__()
        self.N_train = N_train
        self.jitter = jitter
        self.dtype = dtype
        self.device = device
        self.Task = task

        # inducing points
        if fixed_inducing_points:
            self.inducing_index_points = torch.tensor(initial_inducing_points, dtype=dtype).to(device)
        else:
            self.inducing_index_points = nn.Parameter(torch.tensor(initial_inducing_points, dtype=dtype).to(device), requires_grad=True)

        self.group_number = group_number

        # length scale of the kernel
        self.kernel = CauchyKernel(scale=kernel_scale, fixed_scale=fixed_gp_params, dtype=dtype, device=device).to(device)
        self.Z_kernel = EQKernel(scale=0.1, fixed_scale=fixed_gp_params, dtype=dtype, device=device).to(device)
        self.BatchedCauchyKernel = BatchedCauchyKernel(scale=[kernel_scale]*group_number, fixed_scale=fixed_gp_params, dtype=dtype,
                                                       device=device).to(device)
        self.sample_kernel = SampleKernel().to(device)

    def reset_initial_inducing_points(self,initial_inducing_points):
        if self.fixed_inducing_points:
            self.inducing_index_points = torch.tensor(initial_inducing_points, dtype=self.dtype).to(self.device)
        else:
            self.inducing_index_points = nn.Parameter(torch.tensor(initial_inducing_points, dtype=self.dtype).to(self.device),
                                                      requires_grad=True)

    def _kernel_matrix(self, x, y, x_inducing=True, y_inducing=True, diag_only=False):
        """
        Computes GP kernel matrix K(x,y).
        :param x:
        :param y:
        :param x_inducing: whether x is a set of inducing points
        :param y_inducing: whether y is a set of inducing points
        :param diag_only: whether or not to only compute diagonal terms of the kernel matrix
        :return:
        """

        if diag_only:
            matrix = self.kernel.forward_diag(x, y)
        else:
            matrix = self.kernel(x, y)
        return matrix
    
    def _kernel_matrix_ND(self, x, y, x_inducing=True, y_inducing=True, diag_only=False):
        """
        Computes GP kernel matrix K(x,y).
        :param x:
        :param y:
        :param x_inducing: whether x is a set of inducing points
        :param y_inducing: whether y is a set of inducing points
        :param diag_only: whether or not to only compute diagonal terms of the kernel matrix
        :return:
        """
        pos_x = x[:, :-1]
        pos_y = y[:, :-1]

        valid_indices_x = ~torch.isnan(x[:, -1])
        valid_indices_y = ~torch.isnan(y[:, -1])

        # 去除 None (NaN) 的索引
        # 根据有效索引选择 z_x 和 z_y
        z_x = x[valid_indices_x, -1].reshape((-1, 1))
        z_y = y[valid_indices_y, -1].reshape((-1, 1))

        if diag_only:
            K_f = self.Z_kernel.forward_diag(z_x, z_y)
            K_x = self.kernel.forward_diag(pos_x, pos_y)
            matrix = K_f * K_x
        else:
            K_x = self.kernel(pos_x, pos_y)
            # 将 Z_kernel 作为任务核 (task kernel K_f)
            K_f = self.Z_kernel(z_x, z_y)  # Task kernel is now Z_kernel
            matrix = K_f * K_x  # Kronecker product of task and position kernels
        return matrix

    def _kernel_matrix_batchgroup(self, x, y, x_inducing=True, y_inducing=True, diag_only=False):
        """
        Computes GP kernel matrix K(x,y).
        :param x:
        :param y:
        :param x_inducing: whether x is a set of inducing points
        :param y_inducing: whether y is a set of inducing points
        :param diag_only: whether or not to only compute diagonal terms of the kernel matrix
        :return:
        """
        pos_x = x[:, :2]
        pos_y = y[:, :2]

        sample_x = x[:, 2:]
        sample_y = y[:, 2:]

        if self.allow_batch_kernel_scale:
            if diag_only:
                #matrix = torch.diagonal(self.sample_kernel(sample_x, sample_y) * self.kernel(pos_x, pos_y))
                matrix_diag_1 = self.sample_kernel.forward_diag(sample_x, sample_y)
                matrix_diag_2 = self.BatchedCauchyKernel.forward_diag(pos_x, pos_y, sample_x, sample_y)
                matrix = matrix_diag_1 * matrix_diag_2
            else:
                matrix = self.sample_kernel(sample_x, sample_y) * self.BatchedCauchyKernel(pos_x, pos_y, sample_x, sample_y)
        return matrix


    def kernel_matrix(self,  x, y, x_inducing=True, y_inducing=True, diag_only=False):
        if self.Task == 'gptime':
            return self._kernel_matrix_Temporal(x, y, x_inducing, y_inducing, diag_only)
        elif self.Task =='PosSVGP':
            return self._kernel_matrix(x, y, x_inducing, y_inducing, diag_only)
        elif self.Task == 'time':
            return self._kernel_matrix_Temporal(x, y, x_inducing, y_inducing, diag_only)
        elif self.Task == 'BatchGroup':
            return self._kernel_matrix_batchgroup(x, y, x_inducing, y_inducing, diag_only)
        elif 'RZ' in self.Task or 'RT' in self.Task:
            return self._kernel_matrix_ND(x, y, x_inducing, y_inducing, diag_only)
        elif '3D' in self.Task or '4D' in self.Task:
            return self._kernel_matrix(x, y, x_inducing, y_inducing, diag_only)
        else:  # self.Task in [None,'batch' ,'align' ,'3D', 'image', 'ComplexTrans',...]
            return self._kernel_matrix(x, y, x_inducing, y_inducing, diag_only)

    def variational_loss(self, x, y, noise, mu_hat, A_hat):
        """
        Computes L_H for the data in the current batch.
        :param x: auxiliary data for current batch (batch, 1 + 1 + M)
        :param y: mean vector for current latent channel, output of the encoder network (batch, 1)
        :param noise: variance vector for current latent channel, output of the encoder network (batch, 1)
        :param mu_hat:
        :param A_hat:
        :return: sum_term, KL_term (variational loss = sum_term + KL_term)  (1,)
        """
        b = x.shape[0]
        m = self.inducing_index_points.shape[0]

        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points) # (m,m)
        K_mm_inv = torch.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter)) # (m,m)

        K_nn = self.kernel_matrix(x, x, x_inducing=False, y_inducing=False, diag_only=True) # (b)

        K_nm = self.kernel_matrix(x, self.inducing_index_points, x_inducing=False)  # (b, m)
        K_mn = torch.transpose(K_nm, 0, 1)

#        S = A_hat

        # KL term
        mean_vector = torch.matmul(K_nm, torch.matmul(K_mm_inv, mu_hat))

        K_mm_chol = torch.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter))
        S_chol = torch.linalg.cholesky(_add_diagonal_jitter(A_hat, self.jitter))
        K_mm_log_det = 2 * torch.sum(torch.log(torch.diagonal(K_mm_chol)))
        S_log_det = 2 * torch.sum(torch.log(torch.diagonal(S_chol)))

        KL_term = 0.5 * (K_mm_log_det - S_log_det - m +
                             torch.trace(torch.matmul(K_mm_inv, A_hat)) +
                             torch.sum(mu_hat * torch.matmul(K_mm_inv, mu_hat)))

        # diag(K_tilde), (b, )
        precision = 1 / noise

        K_tilde_terms = precision * (K_nn - torch.diagonal(torch.matmul(K_nm, torch.matmul(K_mm_inv, K_mn))))

        # k_i \cdot k_i^T, (b, m, m)
        lambda_mat = torch.matmul(K_nm.unsqueeze(2), torch.transpose(K_nm.unsqueeze(2), 1, 2))

        # K_mm_inv \cdot k_i \cdot k_i^T \cdot K_mm_inv, (b, m, m)
        lambda_mat = torch.matmul(K_mm_inv, torch.matmul(lambda_mat, K_mm_inv))

        # Trace terms, (b,)
        trace_terms = precision * torch.einsum('bii->b', torch.matmul(A_hat, lambda_mat))

        # L_3 sum part, (1,)
        L_3_sum_term = -0.5 * (torch.sum(K_tilde_terms) + torch.sum(trace_terms) +
                                torch.sum(torch.log(noise)) + b * np.log(2 * np.pi) +
                                torch.sum(precision * (y - mean_vector) ** 2))

        return L_3_sum_term, KL_term

    def approximate_posterior_params(self, index_points_test, index_points_train=None, y=None, noise=None):
        """
        Computes parameters of q_S.
        :param index_points_test: X_*
        :param index_points_train: X_Train
        :param y: y vector of latent GP
        :param noise: noise vector of latent GP
        :return: posterior mean at index points,
                 (diagonal of) posterior covariance matrix at index points
        """
        b = index_points_train.shape[0]

        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points) # (m,m)
        K_mm_inv = torch.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter)) # (m,m)

        K_xx = self.kernel_matrix(index_points_test, index_points_test, x_inducing=False,
                                  y_inducing=False, diag_only=True)  # (x)
        K_xm = self.kernel_matrix(index_points_test, self.inducing_index_points, x_inducing=False)  # (x, m)
        K_mx = torch.transpose(K_xm, 0, 1)  # (m, x)

        K_nm = self.kernel_matrix(index_points_train, self.inducing_index_points, x_inducing=False)  # (N, m)
        K_mn = torch.transpose(K_nm, 0, 1)  # (m, N)

        sigma_l = K_mm + (self.N_train / b) * torch.matmul(K_mn, K_nm / noise[:,None])
        sigma_l_inv = torch.linalg.inv(_add_diagonal_jitter(sigma_l, self.jitter))
        mean_vector = (self.N_train / b) * torch.matmul(K_xm, torch.matmul(sigma_l_inv, torch.matmul(K_mn, y/noise)))

        K_xm_Sigma_l_K_mx = torch.matmul(K_xm, torch.matmul(sigma_l_inv, K_mx))
        B = K_xx + torch.diagonal(-torch.matmul(K_xm, torch.matmul(K_mm_inv, K_mx)) + K_xm_Sigma_l_K_mx)

        mu_hat = (self.N_train / b) * torch.matmul(torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mn)), y / noise)
        A_hat = torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mm))

        return mean_vector, B, mu_hat, A_hat

    def kernel_matrix_cached(self):
        self.K_mm = self.kernel_matrix(self.inducing_index_points,self.inducing_index_points)
        self.K_mm_inv = self.stable_inv(self.K_mm)

    def stable_inv(self, matrix):
        jitter_matrix = self.jitter * torch.eye(matrix.size(-1), device=matrix.device)
        chol = torch.linalg.cholesky(matrix + jitter_matrix)
        return torch.cholesky_inverse(chol)

    def compute_variational_and_posterior(self,
                                          index_points_train,
                                          y,
                                          noise,
                                          l,
                                          index_points_test=None,
                                          training=True,
                                          batch_onehot = None):
        """
        Computes variational loss and posterior parameters in a single pass.
        Returns:
        - variational_loss (L_3_sum_term, KL_term)
        - posterior parameters (mean_vector, B)
        """
        # ensure the shape of y and noise is [b]
        y = y.squeeze()  # ensure y is [b]
        noise = noise.squeeze()  # ensure noise is [b]

        # update the cache (only when needed)
        self.kernel_matrix_cached()

        b = index_points_train.shape[0]
        m = self.inducing_index_points.shape[0]

        # compute the kernel matrix
        K_nm = self.kernel_matrix(index_points_train, self.inducing_index_points, x_inducing=False)  # [b, m]
        K_mn = K_nm.T  # [m, b]
        K_nn = self.kernel_matrix(index_points_train, index_points_train,
                                  x_inducing=False, y_inducing=False, diag_only=True)  # [b]

        # use the Woodbury matrix identity to optimize the computation
        scaled_K_nm = K_nm / noise[:, None]  # [b, m]

        # choose the optimal computation method based on the matrix size
        if m > 2 * b:  # when the number of inducing points is much larger than the batch size, use the Woodbury identity
            U = K_mn
            V = (self.N_train / b) * scaled_K_nm
            I_b = torch.eye(b, device=self.device)
            middle_term = I_b + V @ self.K_mm_inv @ U
            inv_term = torch.linalg.inv(middle_term)
            sigma_l_inv = self.K_mm_inv - (self.K_mm_inv @ U) @ inv_term @ (V @ self.K_mm_inv)
        else:
            sigma_l = self.K_mm + (self.N_train / b) * K_mn @ scaled_K_nm
            sigma_l_inv = self.stable_inv(sigma_l)

        # compute mu_hat (the mean of the variational distribution)
        y_scaled = y / noise  # [b]
        mu_hat = (self.N_train / b) * self.K_mm @ sigma_l_inv @ K_mn @ y_scaled  # [m]
        mu_hat = mu_hat.unsqueeze(1)  # [m, 1] keep the dimension consistent

        # compute A_hat (the covariance of the variational distribution)

        if training:
            # ========== training mode: compute the variational loss ==========
            A_hat = self.K_mm @ sigma_l_inv @ self.K_mm  # [m, m]
            # compute the mean of the current batch
            mean_vector = K_nm @ (self.K_mm_inv @ mu_hat).squeeze()  # [b]

            # KL divergence term
            K_mm_chol = torch.linalg.cholesky(_add_diagonal_jitter(self.K_mm, self.jitter))
            S_chol = torch.linalg.cholesky(_add_diagonal_jitter(A_hat, self.jitter))

            logdet_K_mm = 2 * torch.sum(torch.log(torch.diagonal(K_mm_chol)))
            logdet_S = 2 * torch.sum(torch.log(torch.diagonal(S_chol)))
            trace_term = torch.trace(self.K_mm_inv @ A_hat)
            mu_term = mu_hat.squeeze() @ self.K_mm_inv @ mu_hat.squeeze()

            KL_term = 0.5 * (logdet_K_mm - logdet_S - m + trace_term + mu_term)

            # likelihood term (L_3_sum_term)
            # compute the K_tilde term
            K_tilde_terms = (1 / noise) * (K_nn - torch.sum(K_nm * (self.K_mm_inv @ K_nm.T).T, dim=1))

            # compute the trace term
            M = self.K_mm_inv @ A_hat @ self.K_mm_inv
            trace_terms = (1 / noise) * torch.sum((K_nm @ M) * K_nm, dim=1)

            K_xm = K_nm  # Reusing K_nm as test and train are the same
            K_mx = K_xm.T  # (m, b)
            K_xx = K_nn  # Reusing K_nn as diagonal-only
            K_xm_Sigma_l_K_mx = torch.matmul(K_xm, torch.matmul(sigma_l_inv, K_mx))
            K_xm_K_mm_inv = torch.matmul(K_xm, self.K_mm_inv)  # (b, m)
            K_xm_term = torch.matmul(K_xm_K_mm_inv, K_mx)  # (b, b)
            B = K_xx + torch.diagonal(-K_xm_term + K_xm_Sigma_l_K_mx, dim1=0, dim2=1)  # (b,)

            # combine all terms
            L_3_sum_term = -0.5 * (
                    torch.sum(K_tilde_terms) +
                    torch.sum(trace_terms) +
                    torch.sum(torch.log(noise)) +
                    b * np.log(2 * np.pi) +
                    torch.sum((1 / noise) * (y - mean_vector) ** 2)
            )

            variational_loss = (L_3_sum_term, KL_term)
            posterior_params = (mean_vector, B)

        else:
            # ========== prediction mode: compute the posterior parameters ==========
            if index_points_test is None:
                index_points_test = index_points_train

            # compute the kernel matrix of the test points
            K_xm = self.kernel_matrix(index_points_test, self.inducing_index_points, x_inducing=False)  # [t, m]
            K_xx = self.kernel_matrix(index_points_test, index_points_test,
                                      x_inducing=False, y_inducing=False, diag_only=True)  # [t]

            # compute the mean of the test points
            mean_vector = K_xm @ (self.K_mm_inv @ mu_hat).squeeze()  # [t]
            # compute the variance of the test points (only the diagonal)
            K_xm_Sigma_l_K_mx = K_xm @ sigma_l_inv @ K_xm.T
            K_xm_K_mm_inv = K_xm @ self.K_mm_inv
            K_xm_term = K_xm_K_mm_inv @ K_xm.T
            B = K_xx + torch.diag(-K_xm_term + K_xm_Sigma_l_K_mx)  # [t]

            variational_loss = (None, None)
            posterior_params = (mean_vector, B)

        # return the results
        return variational_loss, posterior_params

class PosSVGP(SVGP):
    def __init__(self, fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, jitter, N_train, dtype, device, Task="PosSVGP", **kwargs):
        super(PosSVGP, self).__init__(fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, jitter, N_train, dtype, device, Task, **kwargs)
        self.SVGP_name = 'PosSVGP'

    def compute_variational_and_posterior_transformation(self,
                                          index_points_train,
                                          index_points_train_transformation,
                                          y,
                                          noise,
                                          l,
                                          training=True,
                                          batch_onehot=None):
        """
        Computes variational loss and posterior parameters in a single pass.
        Returns:
        - variational_loss (L_3_sum_term, KL_term)
        - posterior parameters (mean_vector, B)
        """

        b = index_points_train.shape[0]
        m = self.inducing_index_points.shape[0]
        # Kernel matrices
        self.kernel_matrix_cached()
        K_nm = self.kernel_matrix(index_points_train, self.inducing_index_points, x_inducing=False)  # (b, m)
        K_mn = K_nm.T  # (m, b)
        K_nn = self.kernel_matrix(index_points_train, index_points_train, x_inducing=False, y_inducing=False,
                                  diag_only=True)  # (b)

        # Sigma_l and its inverse
        scaled_K_nm = K_nm / noise[:, None]  # 先归一化
        if m > 2 * b:  # 当诱导点数远大于batch size时触发
            U = K_mn  # (m, b)
            V = (self.N_train / b) * scaled_K_nm  # (b, m)
            I_b = torch.eye(b, device=self.device)
            inv_term = torch.inverse(I_b + V @ self.K_mm_inv @ U)  # (b,b)
            sigma_l_inv = self.K_mm_inv - (self.K_mm_inv @ U) @ inv_term @ (V @ self.K_mm_inv)
        else:
            sigma_l = self.K_mm + (self.N_train / b) * torch.matmul(K_mn, scaled_K_nm)  # 避免重复广播
            sigma_l_inv = self.stable_inv(sigma_l)

        # Mu_hat
        mu_hat = (self.N_train / b) * torch.matmul(torch.matmul(self.K_mm, sigma_l_inv),
                                                   torch.matmul(K_mn, y / noise))  # (m, 1)

        # Mean vector
        mean_vector = index_points_train_transformation[:, l] + torch.matmul(K_nm, torch.matmul(self.K_mm_inv,
                                                                                                mu_hat))  # (b, )
        # Posterior covariance
        K_xm = K_nm  # Reusing K_nm as test and train are the same
        K_mx = K_xm.T  # (m, b)
        K_xx = K_nn  # Reusing K_nn as diagonal-only
        K_xm_Sigma_l_K_mx = torch.matmul(K_xm, torch.matmul(sigma_l_inv, K_mx))

        K_xm_K_mm_inv = torch.matmul(K_xm, self.K_mm_inv)  # (b, m)
        K_xm_term = torch.matmul(K_xm_K_mm_inv, K_mx)  # (b, b)
        B = K_xx + torch.diagonal(-K_xm_term + K_xm_Sigma_l_K_mx, dim1=0, dim2=1)  # (b,)


        if training == False:
            return (None, None), (mean_vector, B)

        # A_hat
        A_hat = torch.matmul(self.K_mm, torch.matmul(sigma_l_inv, self.K_mm))  # (m, m)

        # KL divergence terms
        K_mm_chol = torch.linalg.cholesky(_add_diagonal_jitter(self.K_mm, self.jitter))
        S_chol = torch.linalg.cholesky(_add_diagonal_jitter(A_hat, self.jitter))

        logdet_K_mm = 2 * torch.sum(torch.log(torch.diagonal(K_mm_chol)))  # 使用对角线直接求 log(det)
        logdet_S = 2 * torch.sum(torch.log(torch.diagonal(S_chol)))
        trace_term = torch.einsum('ij,ji->', self.K_mm_inv, A_hat)  # 利用 einsum 优化 trace 计算
        mu_term = torch.vdot(mu_hat.squeeze(), self.K_mm_inv @ mu_hat.squeeze())
        KL_term = 0.5 * (logdet_K_mm - logdet_S - m + trace_term + mu_term)

        # Variational loss terms
        K_tilde_terms = (1 / noise) * (
                K_nn - torch.sum(K_nm * (self.K_mm_inv @ K_nm.T).T, dim=1)  # 直接利用 sum 替代 diagonal
        )

        M = self.K_mm_inv @ A_hat @ self.K_mm_inv  # (m, m)
        temp = torch.matmul(K_nm, M)  # (b, m)
        term_per_sample = torch.sum(temp * K_nm, dim=1)  # (b,)
        trace_terms = (1 / noise) * term_per_sample

        L_3_sum_term = -0.5 * (
                torch.sum(K_tilde_terms) + torch.sum(trace_terms) +
                torch.sum(torch.log(noise)) + b * np.log(2 * np.pi) +
                torch.sum((1 / noise) * (y - mean_vector) ** 2)
        )
        return (L_3_sum_term, KL_term), (mean_vector, B)

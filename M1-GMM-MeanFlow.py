import torch
import numpy as np
from sklearn.mixture import GaussianMixture


class DiffusionMeanFlow:
    """MeanFlow (Geng et al., 2025): average-velocity training and Eq.12 sampling."""

    def __init__(self, noise_steps=100, data_length=1024, device=None, gmm_components=5):
        self.noise_steps = int(noise_steps)
        self.data_length = int(data_length)
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.gmm_components = int(gmm_components)
        self.gmm_max_components = 8
        self.gmm_reg_covar = 1e-5
        self.gmm_fitted = False

        self.gmm_means_np = None
        self.gmm_covariances_np = None
        self.gmm_weights_np = None
        self.gmm_data_mean_np = None
        self.gmm_data_std_np = None

        self.gmm_means = None
        self.gmm_covariances = None
        self.gmm_weights = None
        self.gmm_data_mean = None
        self.gmm_data_std = None

        self.time_dist = 'uniform'   # 'uniform' | 'lognorm'
        self.time_mu = -0.4
        self.time_sigma = 1.0
        self.r_eq_t_ratio = 0.50
        self.jvp_api = 'func'        # 'autograd' | 'func' | 'finite_diff'
        self.fd_eps = 1e-3
        self.time_condition_mode = 'dual'  # 'delta_t' | 'dual'

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _sample_r_t(self, batch_size, device, r_eq_t_ratio=None, time_dist=None, time_mu=None, time_sigma=None):
        dist = time_dist or self.time_dist
        mu = self.time_mu if time_mu is None else float(time_mu)
        sigma = self.time_sigma if time_sigma is None else float(time_sigma)

        if dist == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif dist == 'lognorm':
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = self._sigmoid(normal_samples).astype(np.float32)
        else:
            raise ValueError("time_dist must be 'uniform' or 'lognorm'")

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        ratio = self.r_eq_t_ratio if r_eq_t_ratio is None else float(r_eq_t_ratio)
        ratio = max(0.0, min(1.0, ratio))
        num_eq = int(ratio * batch_size)
        if num_eq > 0:
            indices = np.random.permutation(batch_size)[:num_eq]
            r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return r, t

    def _predict_u(self, model, z_in, r_in, t_in):#u = self._predict_u(model, xt, r_vec, t_vec)
        dt = torch.clamp(t_in - r_in, min=0.0, max=1.0)
        if self.time_condition_mode == 'delta_t':
            return model(z_in, dt)
        if self.time_condition_mode == 'dual':
            u_t = model(z_in, torch.clamp(t_in, min=0.0, max=1.0))
            u_dt = model(z_in, dt)
            return 0.5 * (u_t + u_dt)
        raise ValueError("time_condition_mode must be 'delta_t' or 'dual'")

    def fit_gmm(self, data_samples):
        if isinstance(data_samples, torch.Tensor):
            data_samples = data_samples.detach().cpu().numpy()

        data_samples = np.asarray(data_samples)
        if data_samples.ndim == 3:
            n_samples, _, data_length = data_samples.shape
            data_flat = data_samples.reshape(n_samples, data_length)
        elif data_samples.ndim == 2:
            n_samples, data_length = data_samples.shape
            data_flat = data_samples
        else:
            raise ValueError("data_samples must have shape (N, 1, L) or (N, L)")

        if n_samples < 2:
            raise ValueError("Need at least 2 samples to fit GMM")

        data_flat = data_flat.astype(np.float64)
        self.gmm_data_mean_np = data_flat.mean(axis=0, keepdims=True)
        self.gmm_data_std_np = np.maximum(data_flat.std(axis=0, keepdims=True), 1e-6)
        data_norm = (data_flat - self.gmm_data_mean_np) / self.gmm_data_std_np

        max_components = int(max(1, min(self.gmm_max_components, n_samples)))
        best_bic = np.inf
        best_gmm = None

        for n_components in range(1, max_components + 1):
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='diag',
                    init_params='random',
                    n_init=10,
                    max_iter=300,
                    random_state=42,
                    reg_covar=self.gmm_reg_covar,
                    tol=1e-4,
                )
                gmm.fit(data_norm)
                bic = gmm.bic(data_norm)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except Exception:
                continue

        if best_gmm is None:
            n_components = int(max(1, min(self.gmm_components, n_samples)))
            best_gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='diag',
                init_params='random',
                n_init=10,
                max_iter=300,
                random_state=42,
                reg_covar=self.gmm_reg_covar,
                tol=1e-4,
            )
            best_gmm.fit(data_norm)

        self.gmm_components = int(best_gmm.weights_.shape[0])
        self.gmm_means_np = best_gmm.means_.astype(np.float64)
        self.gmm_covariances_np = best_gmm.covariances_.astype(np.float64)
        self.gmm_weights_np = best_gmm.weights_.astype(np.float64)

        self.gmm_data_mean = torch.from_numpy(self.gmm_data_mean_np.astype(np.float32)).to(self.device)
        self.gmm_data_std = torch.from_numpy(self.gmm_data_std_np.astype(np.float32)).to(self.device)
        self.gmm_means = torch.from_numpy(self.gmm_means_np.astype(np.float32)).to(self.device)
        self.gmm_covariances = torch.from_numpy(self.gmm_covariances_np.astype(np.float32)).to(self.device)
        self.gmm_weights = torch.from_numpy(self.gmm_weights_np.astype(np.float32)).to(self.device)
        self.gmm_fitted = True

        print(f"  GMM fitted: components={self.gmm_components}, BIC={best_bic:.2f}")

    def save_gmm_params(self, save_path):
        if not self.gmm_fitted:
            raise RuntimeError("GMM is not fitted, cannot save parameters")
        np.savez(
            save_path,
            means=self.gmm_means_np,
            covariances=self.gmm_covariances_np,
            weights=self.gmm_weights_np,
            data_mean=self.gmm_data_mean_np,
            data_std=self.gmm_data_std_np,
        )

    def load_gmm_params(self, load_path):
        params = np.load(load_path)
        self.gmm_means_np = params["means"].astype(np.float64)
        self.gmm_covariances_np = params["covariances"].astype(np.float64)
        self.gmm_weights_np = params["weights"].astype(np.float64)

        if "data_mean" in params and "data_std" in params:
            self.gmm_data_mean_np = params["data_mean"].astype(np.float64)
            self.gmm_data_std_np = params["data_std"].astype(np.float64)
        else:
            self.gmm_data_mean_np = np.zeros((1, self.gmm_means_np.shape[1]), dtype=np.float64)
            self.gmm_data_std_np = np.ones((1, self.gmm_means_np.shape[1]), dtype=np.float64)

        self.gmm_components = int(self.gmm_weights_np.shape[0])

        self.gmm_data_mean = torch.from_numpy(self.gmm_data_mean_np.astype(np.float32)).to(self.device)
        self.gmm_data_std = torch.from_numpy(self.gmm_data_std_np.astype(np.float32)).to(self.device)
        self.gmm_means = torch.from_numpy(self.gmm_means_np.astype(np.float32)).to(self.device)
        self.gmm_covariances = torch.from_numpy(self.gmm_covariances_np.astype(np.float32)).to(self.device)
        self.gmm_weights = torch.from_numpy(self.gmm_weights_np.astype(np.float32)).to(self.device)
        self.gmm_fitted = True

    def _sample_gmm_noise(self, shape, device):
        batch_size, _, data_length = shape

        if not self.gmm_fitted:
            return torch.randn((batch_size, 1, data_length), device=device)

        component_idx = np.random.choice(
            self.gmm_components,
            size=batch_size,
            p=self.gmm_weights_np,
        )

        noise_samples = []
        for idx in component_idx:
            mean = self.gmm_means[idx].to(device)
            var = self.gmm_covariances[idx].to(device)
            std = torch.sqrt(var + 1e-8)
            eps = torch.randn(data_length, device=device)
            sample_norm = mean + std * eps
            sample = sample_norm * self.gmm_data_std.squeeze(0).to(device) + self.gmm_data_mean.squeeze(0).to(device)
            noise_samples.append(sample)

        return torch.stack(noise_samples, dim=0).unsqueeze(1)

    def prepare_meanflow_batch(self, x0, t):
        t_b = t.view(-1, 1, 1)
        x1 = self._sample_gmm_noise(x0.shape, x0.device)
        xt = (1.0 - t_b) * x0 + t_b * x1
        return xt, x0, x1

    def noise_data(self, x0, t):
        return self.prepare_meanflow_batch(x0, t)

    def meanflow_loss(self, model, x0, r_eq_t_ratio=None, loss_p=1.0, loss_c=1e-3, time_dist=None, time_mu=None, time_sigma=None):
        batch_size = x0.shape[0]
        r, t = self._sample_r_t(
            batch_size,
            x0.device,
            r_eq_t_ratio=r_eq_t_ratio,
            time_dist=time_dist,
            time_mu=time_mu,
            time_sigma=time_sigma,
        )

        t_b = t.view(-1, 1, 1)
        x1 = self._sample_gmm_noise(x0.shape, x0.device)
        z_t = (1.0 - t_b) * x0 + t_b * x1
        v_t = x1 - x0

        def u_fn(z_in, r_in, t_in):
            return self._predict_u(model, z_in, r_in, t_in)

        tangent = (v_t, torch.zeros_like(r), torch.ones_like(t))
        has_func_jvp = hasattr(torch, 'func') and hasattr(torch.func, 'jvp')

        def finite_diff_du_dt(eps):
            u_base = u_fn(z_t, r, t)
            z_eps = z_t + eps * v_t
            t_eps = torch.clamp(t + eps, min=0.0, max=1.0)
            u_eps = u_fn(z_eps, r, t_eps)
            du = (u_eps - u_base) / eps
            return u_base, du

        if self.jvp_api == 'finite_diff':
            u_pred, du_dt = finite_diff_du_dt(self.fd_eps)
        elif self.jvp_api == 'func' and has_func_jvp:
            try:
                u_pred, du_dt = torch.func.jvp(u_fn, (z_t, r, t), tangent, strict=False)
            except Exception:
                u_pred, du_dt = finite_diff_du_dt(self.fd_eps)
        else:
            try:
                u_pred, du_dt = torch.autograd.functional.jvp(
                    u_fn,
                    (z_t, r, t),
                    tangent,
                    create_graph=True,
                )
            except Exception:
                if has_func_jvp:
                    try:
                        u_pred, du_dt = torch.func.jvp(u_fn, (z_t, r, t), tangent, strict=False)
                    except Exception:
                        u_pred, du_dt = finite_diff_du_dt(self.fd_eps)
                else:
                    u_pred, du_dt = finite_diff_du_dt(self.fd_eps)

        delta_t_b = (t - r).view(-1, 1, 1)
        u_tgt = v_t - delta_t_b * du_dt
        err = u_pred - u_tgt.detach()

        err_sq = err.pow(2).flatten(1).mean(dim=1)
        if loss_p > 0:
            w = (1.0 / (err_sq + loss_c).pow(loss_p)).detach()
            loss = (w * err_sq).mean()
        else:
            loss = err_sq.mean()

        return {
            "loss": loss,
            "u_mse": err_sq.mean().detach(),
            "delta_t_mean": (t - r).mean().detach(),
            "r_eq_t_ratio": (r == t).float().mean().detach(),
        }

    @staticmethod
    def _build_time_grid(steps):
        return torch.linspace(1.0, 0.0, steps + 1)
    
    def sample(
        self,
        model,
        n,
        steps=None,
        sampler="meanflow",
        **kwargs,
    ):
        model.eval()
        steps = max(1, int(steps if steps is not None else self.noise_steps))
        use_amp = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                xt = self._sample_gmm_noise((n, 1, self.data_length), self.device)
                time_grid = self._build_time_grid(steps).to(self.device)

                for i in range(steps):
                    ti = float(time_grid[i].item())#ti=1
                    ri = float(time_grid[i + 1].item())#ri=0
                    delta_t = max(0.0, ti - ri)#1
                    t_vec = torch.full((n,), ti, device=self.device)
                    r_vec = torch.full((n,), ri, device=self.device)
                    u = self._predict_u(model, xt, r_vec, t_vec)
                    xt = xt - delta_t * u#delta_t=1，xt=xt-u

        model.train()
        return xt

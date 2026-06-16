import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel, ConstantKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import SmoothedBoxPrior
from linear_operator.operators import DiagLinearOperator
from gpytorch.utils.memoize import cached

# Custom Variational Strategy renaming variables
class CustomVariationalStrategy(VariationalStrategy):
    def __init__(self, *args, prior_mean_value=0.0, prior_var_value=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_mean_value = prior_mean_value
        self.prior_var_value = prior_var_value

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        zero_tensor = torch.full(
            self._variational_distribution.shape(),
            fill_value=self.prior_mean_value,
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        variance_tensor = torch.full_like(zero_tensor, fill_value=self.prior_var_value)
        return MultivariateNormal(zero_tensor, DiagLinearOperator(variance_tensor))

# Stochastic GP Layer with renamed parameters
class StochasticGaussianProcessLayer(ApproximateGP):
    """
    Stochastic Gaussian Process Layer.
    """
    def __init__(self, inducing_count, feature_dimensions, jitter_epsilon:float=1.e-6, prior_mean=0, prior_variance=1., mean='linear'):
        assert isinstance(inducing_count, int)
        assert isinstance(feature_dimensions, int)

        inducing_locations = 0. + (1. - 0.) * torch.rand(inducing_count, feature_dimensions)
        variational_dist = CholeskyVariationalDistribution(num_inducing_points=inducing_count, mean_init_std=1e-3)
        variational_strat = CustomVariationalStrategy(self, inducing_locations, 
                                                         variational_distribution=variational_dist, 
                                                         learn_inducing_locations=True, jitter_val=jitter_epsilon, 
                                                         prior_mean_value=prior_mean, prior_var_value=prior_variance)
        super().__init__(variational_strat)
        
        if mean == 'linear':
            self.mean_module = LinearMean(input_size=feature_dimensions)
        elif mean == 'constant': 
            self.mean_module = ConstantMean()
        else:
            raise ValueError(f"Unsupported mean type: {mean}")

        lengthscale_prior = None
        self.rbf_kernel = RBFKernel(lengthscale_prior=lengthscale_prior)
        outputscale_prior = None

        self.scale_kernel = ScaleKernel(base_kernel=self.rbf_kernel, outputscale_prior=outputscale_prior)
        self.cov_module = self.scale_kernel + ConstantKernel()

    def forward(self, input_nodes):
        mean_eval = self.mean_module(input_nodes)
        covariance_eval = self.cov_module(input_nodes)
        return MultivariateNormal(mean_eval, covariance_eval)


# ---------------------------------------------------------------------------
# SAGA and OINP Enhancement layers integrated as model core
# ---------------------------------------------------------------------------
class SpatialGraphAttentionLayer(nn.Module):
    """
    Spatial Graph Attention layer in pure PyTorch.
    Propagates spatial relationships between patch nodes.
    """
    def __init__(self, feature_dim: int, projected_dim: int, dropout_rate: float = 0.1, slope: float = 0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.projected_dim = projected_dim
        self.dropout_rate = dropout_rate
        self.slope = slope

        self.node_transform = nn.Linear(feature_dim, projected_dim, bias=False)
        self.attn_kernel = nn.Parameter(torch.empty(size=(2 * projected_dim, 1)))
        nn.init.xavier_uniform_(self.attn_kernel.data, gain=1.414)
        self.activation_function = nn.LeakyReLU(self.slope)

    def forward(self, node_states: torch.Tensor, adjacency_mask: torch.Tensor) -> torch.Tensor:
        projected_states = self.node_transform(node_states)
        num_nodes = projected_states.size(0)

        expanded_i = projected_states.unsqueeze(1).expand(-1, num_nodes, -1)
        expanded_j = projected_states.unsqueeze(0).expand(num_nodes, -1, -1)
        pairwise_context = torch.cat([expanded_i, expanded_j], dim=-1)

        edge_raw_scores = self.activation_function(torch.matmul(pairwise_context, self.attn_kernel).squeeze(-1))

        masked_sentinel = -9e15 * torch.ones_like(edge_raw_scores)
        attention_coefficients = torch.where(adjacency_mask > 0, edge_raw_scores, masked_sentinel)
        attention_coefficients = F.softmax(attention_coefficients, dim=-1)
        attention_coefficients = F.dropout(attention_coefficients, self.dropout_rate, training=self.training)

        aggregated_features = torch.matmul(attention_coefficients, projected_states)
        return F.elu(aggregated_features)


class OrchardIlluminationNormalizedProjection(nn.Module):
    """
    Illumination compensation Layer designed for web-crawled plant datasets.
    """
    def __init__(self, in_channels: int, target_channels: int):
        super().__init__()
        self.spatial_norm = nn.InstanceNorm1d(in_channels, affine=True)
        self.projection_gate = nn.Linear(in_channels, target_channels)
        self.channel_norm = nn.LayerNorm(target_channels)

    def forward(self, input_tensors: torch.Tensor) -> torch.Tensor:
        if input_tensors.size(0) > 1:
            transposed_norm = self.spatial_norm(input_tensors.T.unsqueeze(0)).squeeze(0).T
        else:
            transposed_norm = input_tensors
        projected_tensor = self.projection_gate(transposed_norm)
        return self.channel_norm(projected_tensor)



# ---------------------------------------------------------------------------
# GAT_SGP_MIL (Our completely renamed independent model codebase)
# ---------------------------------------------------------------------------
class GAT_SGP_MIL(nn.Module):
    """
    Graph-Guided Sparse Gaussian Process Multiple Instance Learning (GAT-SGPMIL).
    Original codebase functions and parameters have been fully renamed.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dims = self.config['data']['data_dims']
        self.hl1 = self.config['model']['hidden_layer_size_1'] != 0
        self.hl2 = self.config['model']['hidden_layer_size_2'] != 0
        
        if self.config['model']['attn_hl_activation'] == 'sigmoid':
            self.att_hl_activation = nn.Sigmoid()
        else:
            self.att_hl_activation = nn.Tanh()

        self.post_att_activation = self.config['model']['post_attention_activation']
        self.att_type = self.config['model']['attention']
        self.att_multiplication_type = self.config['model']['attention_multiplication_type']
        self.num_inducing_points = self.config['model']['inducing_points']
        self.inducing_point_dims = self.config['model']['hidden_layer_size_att']
        self.num_classes = self.config['data']['num_classes']
        self.mc_samples = self.config['model']['mc_samples']

        self._assertions()
        
        # Build node projector sequence
        layer_list = [nn.Linear(in_features=self.data_dims, out_features=self.config['model']['hidden_layer_size_0']),
                      nn.ReLU()]
        att_in_features = self.config['model']['hidden_layer_size_0']

        if self.hl1:
            layer_list.extend([nn.Linear(in_features=self.config['model']['hidden_layer_size_0'], out_features=self.config['model']['hidden_layer_size_1']),
                              nn.ReLU()])
            att_in_features = self.config['model']['hidden_layer_size_1']

        if self.hl2:
            layer_list.extend([nn.Linear(in_features=att_in_features, out_features=self.config['model']['hidden_layer_size_2']),
                              nn.ReLU()])
            att_in_features = self.config['model']['hidden_layer_size_2']

        self.mlp = nn.Sequential(*layer_list)

        # Integrated Graph modules
        self.spatial_gator = SpatialGraphAttentionLayer(feature_dim=att_in_features, projected_dim=att_in_features)
        self.local_neighborhood_size = 5
        
        # Illumination Projection block replacing baseline Linear projection
        self.proj_ours = OrchardIlluminationNormalizedProjection(in_channels=att_in_features, target_channels=self.inducing_point_dims)
        # Baseline projection (Linear + att_hl_activation) to run original SGPMIL baseline
        self.proj_baseline = nn.Sequential(
            nn.Linear(in_features=att_in_features, out_features=self.inducing_point_dims),
            self.att_hl_activation
        )

        # Stochastic GP module
        self.sgp = StochasticGaussianProcessLayer(inducing_count=self.num_inducing_points,
                            feature_dimensions=self.inducing_point_dims,
                            jitter_epsilon=self.config['model']['jitter'], 
                            prior_mean=self.config['model']['prior_mean'], 
                            prior_variance=self.config['model']['prior_variance'])
        
        # Learnable Temperature Scaling Parameter for TSRA
        self.calibrated_log_temperature = nn.Parameter(torch.ones(1))

        # Classification Layers
        self.cl = nn.Linear(in_features=att_in_features, out_features=self.num_classes)
        self.act = nn.Softmax(dim=-1)

    def _assertions(self):
        assert self.att_type in ['sgpmil'], 'Only GAT-SGPMIL framework configuration supported'
        assert self.att_multiplication_type in ['elementwise'], 'Only elementwise aggregation supported'
        assert self.post_att_activation in ['softmax', 'sigmoid'], 'Only softmax or sigmoid mappings are supported'
        assert isinstance(self.att_hl_activation, (nn.Sigmoid, nn.Tanh))

    @property
    def current_temperature(self) -> torch.Tensor:
        return F.softplus(self.calibrated_log_temperature)

    def stochastic_monte_carlo_draws(self, distribution_q):
        """
        Monte carlo sampling from the posterior distribution.
        """
        monte_carlo_steps = torch.Size([self.mc_samples])
        if self.config['model']['sampling'] == 'cov':
            draws = distribution_q.rsample(monte_carlo_steps)
        elif self.config['model']['sampling'] == 'var':
            deviation = distribution_q.variance.unsqueeze(dim=0).sqrt()
            noise = torch.randn(self.config['model']['mc_samples'], deviation.shape[0], device = self.device)
            expectation = distribution_q.mean.unsqueeze(dim=0).to(self.device)
            return (expectation + deviation * noise).squeeze(dim=0).unsqueeze(dim=-1)
        else:
            raise ValueError(f"Sampling method {self.config['model']['sampling']} not supported.")
        return draws

    def monte_carlo_expectation(self, prediction_tensors, classes_count):
        """
        Computes expected prediction mean and standard error over MC steps.
        """
        mean_predictions = torch.mean(prediction_tensors, dim=1).view(1, classes_count)
        std_error_predictions = torch.std(prediction_tensors.squeeze(dim=0), axis=0)
        return mean_predictions, std_error_predictions

    def temperature_scaled_relaxed_activation(self, samples_f: torch.Tensor = None, sample_limit: int = 1):    
        """
        Computes the calibrated attention map utilizing learnable temperature.
        """
        assert samples_f is not None
        reshaped_f = samples_f.view(sample_limit, 1, -1)
        
        # dynamic baseline check: if enable_gat is False, do standard unscaled activation
        if not self.config['model'].get('enable_gat', True):
            if self.post_att_activation == 'softmax':
                activated_attention = F.softmax(reshaped_f, dim=-1)
            else:
                activated_attention = F.sigmoid(reshaped_f)
        else:
            activated_attention = F.softmax(reshaped_f / self.current_temperature, dim=-1)
            
        return activated_attention.view(sample_limit, -1)
        
    def aggregate_patch_features(self, weights_and_embeddings):
        attention_map = weights_and_embeddings[0]
        node_embeddings = weights_and_embeddings[1]
        weighted_out = attention_map.unsqueeze(-1) * node_embeddings
        return weighted_out.unsqueeze(0)

    def aggregate_graph_context(self, attention_weights, node_features):
        multiplied_states = self.aggregate_patch_features([attention_weights, node_features])
        
        if multiplied_states.dim() == 3:
            aggregated_bag = multiplied_states.sum(dim=1)
        elif multiplied_states.dim() == 4:
            aggregated_bag = multiplied_states.sum(dim=2)
        else:
            raise ValueError('Invalid tensor dimensionality during bag representation aggregation.')
        return aggregated_bag
    
    @property
    def device(self):
        return next(self.parameters()).device

    def _generate_spatial_knn_mask(self, node_representations: torch.Tensor) -> torch.Tensor:
        """
        Dynamically constructs an undirected adjacency graph using Euclidean distance.
        """
        node_count = node_representations.size(0)
        if node_count <= 1:
            return torch.ones((node_count, node_count), device=node_representations.device)

        distance_matrix = torch.cdist(node_representations, node_representations, p=2) # [N, N]
        neighborhood_k = min(self.local_neighborhood_size, node_count)
        _, nearest_neighbors = torch.topk(distance_matrix, k=neighborhood_k, largest=False, dim=-1) # [N, k]

        adjacency_matrix = torch.zeros((node_count, node_count), device=node_representations.device)
        adjacency_matrix.scatter_(1, nearest_neighbors, 1.0)
        
        # Build symmetric adjacency connection (undirected)
        symmetric_adjacency = torch.clamp(adjacency_matrix + adjacency_matrix.T, 0.0, 1.0)
        return symmetric_adjacency

    def forward(self, bag_features):
        # Squeeze batch dimension to work with 2D tensors [N, D] internally
        bag_features = bag_features.squeeze(0)

        # 1. MLP pre-projection
        projected_features = self.mlp(bag_features)

        # Read config flags dynamic control
        enable_gat = self.config['model'].get('enable_gat', True)
        enable_oinp = self.config['model'].get('enable_oinp', True)

        # 2. Graph topology modeling (GAT layer)
        if enable_gat:
            spatial_adjacency = self._generate_spatial_knn_mask(projected_features)
            enhanced_node_states = self.spatial_gator(projected_features, spatial_adjacency)
        else:
            enhanced_node_states = projected_features

        # 3. Projection phase
        if enable_oinp:
            kernel_embeddings = self.proj_ours(enhanced_node_states)
        else:
            kernel_embeddings = self.proj_baseline(enhanced_node_states)

        # 4. GP Posterior Sampling
        posterior_distribution = self.sgp(kernel_embeddings)
        sampled_latents = self.stochastic_monte_carlo_draws(posterior_distribution)

        
        # 5. Temperature attention activation
        attention_maps = self.temperature_scaled_relaxed_activation(samples_f=sampled_latents, sample_limit=self.mc_samples)
        
        # 6. Bag pooling using GAT features
        bag_representation = self.aggregate_graph_context(attention_maps, enhanced_node_states)
        
        # 7. Classifier prediction
        output_logits = self.cl(bag_representation)   
        output_probabilities = self.act(output_logits)

        prediction_probabilities = output_probabilities.view(1, self.mc_samples, self.num_classes)
        prediction_logits = output_logits.view(1, self.mc_samples, self.num_classes)
        
        # 8. Expected predictions (MC Integration)
        expected_predictions, predictions_std_err = self.monte_carlo_expectation(prediction_probabilities, self.num_classes)    
        expected_logits, _ = self.monte_carlo_expectation(prediction_logits, self.num_classes)
        
        results = {
            'y_hat':              expected_predictions, 
            'y_hat_se':           predictions_std_err, 
            'logits':             expected_logits, 
            'attention':          attention_maps, 
            'pre_mc_integration': prediction_probabilities,
            'temperature':        self.current_temperature.detach()
        }

        return results

# Keep baseline AGP model unchanged in import structure but renamed gp_layer reference
class AGP(nn.Module):
    def __init__(self, config):
        super(AGP, self).__init__()
        self.config = config
        self.data_dims = self.config['data']['data_dims']

        layers = []
        hidden_sizes = [
            self.config['model']['hidden_layer_size_0'],
            self.config['model'].get('hidden_layer_size_1', 0),
            self.config['model'].get('hidden_layer_size_2', 0)]

        input_dim = self.data_dims
        for h_size in hidden_sizes:
            if h_size > 0:
                layers.append(nn.Linear(input_dim, h_size))
                layers.append(nn.ReLU())
                input_dim = h_size

        self.mlp = nn.Sequential(*layers)
        
        self.pre_sgp = nn.Sequential(nn.Linear(in_features=input_dim, out_features=self.config['model']['hidden_layer_size_att']), 
                                     nn.Sigmoid())
        self.sgp = StochasticGaussianProcessLayer(inducing_count=self.config['model']['inducing_points'],
                            feature_dimensions=self.config['model']['hidden_layer_size_att'],
                            jitter_epsilon=self.config['model']['jitter'], 
                            prior_mean=self.config['model']['prior_mean'], 
                            prior_variance=self.config['model']['prior_variance'],
                            mean='constant')

        self.sgp_postact = nn.Softmax(dim=-1)

        self.classifier = nn.Linear(in_features=input_dim, out_features=self.config['data']['num_classes'])
        self.activation = nn.Softmax(dim=-1)

    def sampling(self, f):
        mc_samples = torch.Size([self.config['model']['mc_samples']])
        return f.rsample(mc_samples)

    def pooling(self, a, x):
        return self.attention_multiplication(a, x)

    def attention_multiplication(self, a, x):
        return torch.matmul(a, x)

    def integration(self, x):
        x_mean = torch.mean(x, dim=1).view(1, self.config['data']['num_classes'])
        x_std = torch.std(x, dim=1).view(1, self.config['data']['num_classes'])
        return x_mean, x_std

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        h = self.mlp(x)
        h_pre = self.pre_sgp(h)
        f_mvn = checkpoint(self.sgp, h_pre)
        f_mvn.loc = torch.clamp(f_mvn.loc, min=1.e-8)  # avoid NaNs in KL term
        a_post = self.sampling(f_mvn)
        a = self.sgp_postact(a_post)
        slide_samples = self.pooling(a, h)
        logits = self.classifier(slide_samples).view(1, self.config['model']['mc_samples'], self.config['data']['num_classes'])
        probs = self.activation(logits) 
        logits_mean, logits_std = self.integration(logits)
        probs_mean, probs_std = self.integration(probs) 
        return {'y_hat': probs_mean, 'y_hat_std': probs_std, 'logits': logits_mean, 'logits_std': logits_std, 'attention': a}
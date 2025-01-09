import torch
from torch import nn
from embedding import PotentialSpinEncoding
from torch.func import vmap


class TQS(nn.Module):
    def __init__(
        self,
        embed_dim,
        max_chain_len,
        num_heads,
        num_layers,
        possible_spins,
        dim_feedforward,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim
        self.max_chain_len = max_chain_len

        self.embedding = PotentialSpinEncoding(embed_dim, max_chain_len)

        self.encoder_mask = torch.triu(
            torch.ones(max_chain_len * 2, max_chain_len * 2) * float("-inf"), diagonal=1
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=num_layers
        )

        self.prob_head = nn.Linear(embed_dim, possible_spins)
        self.phase_head = nn.Linear(embed_dim, 1)

    def forward(self, potentials: torch.Tensor, spins: torch.Tensor):
        # potentials: (seq, batch)
        # spins: (seq, batch)

        seq = self.embedding(potentials, spins)  # (seq, batch, embed_dim)
        seq = self.encoder(seq)  # (seq, batch, embed_dim)

        # Pass probability logits through head and softmax
        prob = self.prob_head(seq)  # (seq, batch, possible_spins) logits
        prob = torch.sigmoid(prob)  # (seq, batch, possible_spins) probabilities

        # Pass phase logits through softsign and then scale by
        # pi to get phase
        phase = self.phase_head(seq)  # (seq, batch, 1) logits
        phase = torch.tanh(phase) * torch.pi  # (seq, batch, 1) phase

        return prob, phase

    def sample_spins(self, initial_potentials, max_length):
        # Allocate a buffer for the sampled basis states
        sampled_spins = torch.zeros(max_length, initial_potentials.size(1))
        batch_size = initial_potentials.size(1)
        batch_remaining_idx = torch.arange(batch_size)

        for i in range(max_length):
            # get P(s_{i+1} | V, s_{1:i}) and phi(s_{i+1} | V, s_{1:i}) distributions
            probs, _ = self.forward(
                initial_potentials[:, batch_remaining_idx],
                sampled_spins[:i, batch_remaining_idx],
            )

            # sample s_{i+1} from P(s_{i+1} | V, s_{1:i})
            last_probs = probs[-1]  # (batch, 2); P(s_{i+1} | V, s_{1:i})
            sampled_spins[i, batch_remaining_idx] = (
                torch.multinomial(last_probs, 1).squeeze().float()
            )

            # a mask with dimension (batch_remaining,); True if we sampled a 1
            newly_completed = sampled_spins[i, batch_remaining_idx] == 1.0

            # mask out the batch_remaining_idx that have been completed
            batch_remaining_idx = batch_remaining_idx[~newly_completed]

        return sampled_spins

    def compute_psi(self, x, V, t):
        """
        Computes a complete complex wavefunction at the basis states provided
        """
        chain_length = x.size(0)
        prob, phases = self.forward(V, x)
        prob = prob[chain_length:, :, 1]  # (seq, batch); P(s=1 | V, x)
        phases = phases[chain_length:].squeeze()
        psi = torch.sqrt(prob) * torch.exp(1j * phases)
        return psi

    def weights_from_sample(self, x):
        """
        Conveniently, we can sum over the batch dimension to get a sample distribution
        for the basis states.

        Parameters:
        x: (seq, batch)

        Returns:
        weights: (seq,); the weights for each basis state
        """
        return torch.mean(x, dim=1)

    def psi_terms(self, x, V, t):
        # TODO what effects does circular shifting have on memory usage?
        x_l = torch.roll(x, 1, dims=0)
        x_r = torch.roll(x, -1, dims=0)
        V_l = torch.roll(V, 1, dims=0)
        V_r = torch.roll(V, -1, dims=0)

        psi_x = self.compute_psi(x, V, t)
        psi_l = self.compute_psi(x_l, V_l, t)
        psi_r = self.compute_psi(x_r, V_r, t)

        return psi_x, psi_l, psi_r

    def E_loc(self, psi_x, psi_l, psi_r, V, t):
        """
        Compute E_loc values across a batch of samples x of dimension (seq, batch)
        and potentials V of dimension (seq, batch).
        """
        return -t * 1 / psi_x * (psi_l + psi_r) + V * psi_x

    def d_ln_P_dtheta(self, psi_x: torch.Tensor):
        """
        Compute the derivative of the probability distribution with respect to model
        parameters. Note that this *can* be done with autodiff; we don't perform
        any sampling.
        """
        batch = psi_x.size(1)
        seq = psi_x.size(0)
        ones_like_params = [torch.ones_like(param) for param in self.parameters()]
        for param in self.parameters():
            first_param = param

        # psi_x: (seq, batch)

        # the problem is that we're not preserving the batch dimension; we end up summing
        # across the batch (sample) dimension prematurely, ruining expectation value
        # calculations down the road

        # ones = torch.ones_like(psi_x, requires_grad=True)

        # grad_sample = lambda psi: torch.tensor(
        #     torch.autograd.grad(
        #         torch.log(psi * psi.conj()),
        #         self.parameters(),
        #         create_graph=True,
        #         grad_outputs=ones,
        #         allow_unused=True,
        #     )
        # )

        get_batched_grad = lambda input: torch.autograd.grad(
            torch.log(psi_x * psi_x.conj()),
            self.parameters(),
            create_graph=True,
            grad_outputs=input,
            allow_unused=True,
        )

        return vmap(get_batched_grad, in_dims=1, randomness="same")(torch.eye(batch))

        # print("psi_x.requires_grad:", psi_x.requires_grad)
        # return vmap(grad_sample, in_dims=1, randomness="same")(psi_x)

        # ideally we want a list of tensors (one for each updateable parameter)
        # where each tensor is of shape (batch, param_dims...)
        # return torch.autograd.grad(
        #     torch.log(psi_x * psi_x.conj()),
        #     self.parameters(),
        #     create_graph=True,
        #     grad_outputs=torch.ones_like(psi_x),
        #     # TODO: why would a preliminary d(output)/d(output) be useful?
        #     allow_unused=True,
        # )

    def gradient(self, x, V, t):
        """
        Compute the gradient of the local energy with respect to the model parameters.
        """
        psi_x, psi_l, psi_r = self.psi_terms(x, V, t)
        E_loc = self.E_loc(psi_x, psi_l, psi_r, V, t)  # (seq, batch)
        d_ln_P_dtheta = self.d_ln_P_dtheta(psi_x)  # (seq, batch, num_params)

        return torch.mean(E_loc * d_ln_P_dtheta, dim=1)

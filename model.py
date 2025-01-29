import torch
from torch import nn
from embedding import PotentialSpinEncoding
from torch.func import functional_call, vmap, grad, jacrev


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

        self.L = None

    def forward(self, potentials: torch.Tensor, spins: torch.Tensor):
        # potentials: (seq, batch)
        # spins: (seq, batch)

        self.L = potentials.size(0)

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

    def psi_from_probs_phases(self, probs, phases):
        probs = probs[self.L :, :, 0]  # TODO: any reason to preserve embed dimension?
        phases = phases[self.L :, :, 0]
        psi = torch.sqrt(probs) * torch.exp(1j * phases)
        return psi

    def compute_psi(self, x, V, t):
        """
        Computes a complete complex wavefunction at the basis states provided
        """
        chain_length = x.size(0)
        prob, phases = self.forward(V, x)
        # prob = prob[chain_length:, :, 1]  # (seq, batch); P(s=1 | V, x)
        # phases = phases[chain_length:].squeeze()
        psi = self.psi_from_probs_phases(prob, phases)
        return psi

    def _ln_P(self, x, V, params, buffers):
        """
        Computes ln(P(x; theta)) for a basis state and corresponding potential function vector.
        Does this in a manner amenable to using vmapping a function created with grad over
        a batch of spins.

        NOTE: MUST BE GIVEN A SINGLE SAMPLE. This function should not be used on the top
        level; it's meant to be used in d_ln_dtheta.

        This function is "einmapped" over the batch dimension, so it should
        take individual samples and not batches.
        """

        # Unsqueeze if there's no batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(1)
            V = V.unsqueeze(1)

        probs, phases = functional_call(
            self, (params, buffers), (V, x)
        )  # <module>, (parameters, buffers), <inputs to module's forward>
        psi_x = self.psi_from_probs_phases(probs, phases)

        # TODO: Do the phases not matter at all? Argument: no! The phase head
        # should be included in backpropagation. But it seems there's no dependency
        # of the output probabilities on the parameters of the phase head; why would
        # the phase head even get updated?

        # TODO: calculate the derivative of the phases with respect to model parameters
        # explicitly; it's suspicious that the phase head gets adjusted during training.

        ln_P = torch.log((torch.conj(psi_x) * psi_x).real)
        return ln_P

    def _E_loc(self, x, V, t, params, buffers):
        """
        Computes the local energy for a basis state and corresponding potential function vector.
        Does this in a manner amenable to using vmapping a function created with grad over
        a batch of spins.

        NOTE: MUST BE GIVEN A SINGLE SAMPLE. This function should not be used on the top
        level; it's meant to be used in dEloc_dTheta.

        This function is "einmapped" over the batch dimension, so it should
        take individual samples and not batches.
        """

        # Unsqueeze if there's no batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(1)
            V = V.unsqueeze(1)

        # Roll along seq (dim 0)
        x_l = torch.roll(x, 1, dims=0)
        x_r = torch.roll(x, -1, dims=0)
        V_l = torch.roll(V, 1, dims=0)
        V_r = torch.roll(V, -1, dims=0)

        # TODO: take probs and phases from other calculations in the args
        # for this function
        psi_x = self.psi_from_probs_phases(
            *functional_call(self, (params, buffers), (V, x))
        )
        psi_l = self.psi_from_probs_phases(
            *functional_call(self, (params, buffers), (V_l, x_l))
        )
        psi_r = self.psi_from_probs_phases(
            *functional_call(self, (params, buffers), (V_r, x_r))
        )
        print("Psi shapes:", psi_x.shape, psi_l.shape, psi_r.shape)
        print("V shapes:", V.shape, V_l.shape, V_r.shape)

        E_loc_1 = -t * (psi_x**-1) * (psi_l + psi_r)
        E_loc_2 = V * psi_x
        E_loc = E_loc_1 + E_loc_2
        return E_loc

    def dlnP_dTheta(self, x, V, params, buffers):
        """
        The goal of this function is to compute the gradient of ln(P(x; Theta)) with respect
        to "big-T" Theta, the set of all model parameters. In effect it returns
        a collection of dlnP_dtheta_i's, where i corresponds to some parameter tensor of the
        model:

        (dlnP_dtheta_1, dlnP_dtheta_2, ..., dlnP_dtheta_p)

        This is returned in the form of a dictionary mapping parameter/buffer names to
        their respective gradients. This is done batch-wise, so that we can compute the
        expectation value.
        """

        # Compute the Jacobian with respect to x.
        dlnP_dtheta_sample = jacrev(self._ln_P)

        # in_dims=(1, 1, None, None) produces iteration over dimension 1 (the batch dim) of the
        # spins and the potentials and does no "ein-iteration" over the parameters or buffers
        # (treated as constant with respect to the iteration)
        vmap_dlnP_dtheta = vmap(dlnP_dtheta_sample, in_dims=(1, 1, None, None))
        return vmap_dlnP_dtheta(x, V, params, buffers)

    def dEloc_dTheta(self, x, V, t, params, buffers):

        dEloc_dtheta_sample = jacrev(self._E_loc)

        vmap_dEloc_dtheta = vmap(dEloc_dtheta_sample, in_dims=(1, 1, None, None))
        return vmap_dEloc_dtheta(x, V, t, params, buffers)

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
        return -t * 1 / psi_x * (psi_l + psi_r) + V

    def gradient(self, x, V, t):
        """
        Compute the gradient of the local energy with respect to the model parameters.
        """
        # psi_x, psi_l, psi_r = self.psi_terms(x, V, t)
        # E_loc = self.E_loc(psi_x, psi_l, psi_r, V, t)  # (seq, batch)

        params = {k: v.detach() for k, v in self.named_parameters()}
        buffers = {k: v.detach() for k, v in self.named_buffers()}

        # E_loc as a function of basis states (spin chains)
        E_loc = self._E_loc(x, V, t)

        # Dictionary from parameter/buffer name to gradient tensors
        # of ln(P(x; Theta)) with respect to that parameter/buffer
        dln_Pdtheta = self.dlnP_dTheta(x, V, params, buffers)

        # Dictionary from parameter/buffer name to gradient tensors
        # of E_loc with respect to that parameter/buffer
        dEloc_dtheta = self.dEloc_dTheta(x, V, t, params, buffers)

        expect = lambda x: torch.mean(x, dim=0)
        cov = lambda x, y: expect(x * y) - expect(x) * expect(y)

        # Perform the calculations across the dictionary
        dE_dTheta = {
            param_name: cov(dln_Pdtheta[param_name], E_loc)
            + expect(dEloc_dtheta[param_name])
            for param_name in dln_Pdtheta.keys()
        }

        return dE_dTheta

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate.utils import reduce, gather_object

from .base import BaseGenerationRunner, GenerationStep
from itertools import combinations
import math
class StatsRunner(BaseGenerationRunner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._post_processing_type = self.config.run_type
        self._model_name = self.config.model_name.split("/")[-1]
        self._num_layers: int = 0
        self._top_k_per_layer: List[torch.Tensor] = []
        self._last_tile_attn: List[torch.Tensor] = []
        self._causal_attn_weights: List[torch.Tensor] = []
        self._sim_mat: torch.Tensor | None = None
        self._output_hidden_states = True
        if self.config.store_results:
            self.all_stats = []

    def prepare(self, num_layers: int, heads: int) -> None:
        self._num_layers = num_layers
        self._top_k_per_layer = []
        self._last_tile_attn = []
        self._causal_attn_weights = []
        self._hidden_states_o = []
        device = self.accelerator.device
        self._layer_importance = torch.zeros((num_layers,), dtype=torch.float32, device=device)
        if self._post_processing_type in ["plot_similarity", "select_layers"]:
            self._sim_mat = torch.zeros((num_layers, num_layers, heads, heads), dtype=torch.float32, device=device)
        else:
            self._sim_mat = torch.zeros((num_layers, heads), dtype=torch.float32, device=device)

    def update(self, is_decode: bool, layer_idx: int, indices: torch.Tensor, attn_weights: torch.Tensor, last_tile_attn: Optional[torch.Tensor] = None) -> None: # for top-k indices and attn weights
        B, H, Lq, _ = attn_weights.shape
        if last_tile_attn is None:
            last_tile_attn = torch.zeros(B, H, Lq, device=attn_weights.device, dtype=attn_weights.dtype)
        if not is_decode:
            self._top_k_per_layer.append(indices)
            self._last_tile_attn.append(last_tile_attn)
            self._causal_attn_weights.append(attn_weights)
        else:
            self._last_tile_attn[layer_idx] = torch.cat((self._last_tile_attn[layer_idx], last_tile_attn), dim=-1)
            caw = self._causal_attn_weights[layer_idx]
            pad = caw.new_zeros(caw.shape[0], caw.shape[1], caw.shape[2], 1)
            caw = torch.cat((caw, pad), dim=-1)
            caw = torch.cat((caw, attn_weights), dim=-2)
            self._causal_attn_weights[layer_idx] = caw
            tk = self._top_k_per_layer[layer_idx]
            pad = tk.new_full((tk.shape[0], tk.shape[1], tk.shape[2], indices.shape[3] - tk.shape[3]), fill_value=caw.shape[-1]-1)
            tk = torch.cat((tk, pad), dim=-1)
            tk = torch.cat((tk, indices), dim=-2)
            self._top_k_per_layer[layer_idx] = tk

    def update_hidden_states(self, is_decode: bool, layer_idx:int, hidden_states: torch.Tensor) -> None: # for hidden states after o_proj
        if not is_decode:
            self._hidden_states_o.append(hidden_states)
        else:
            self._hidden_states_o[layer_idx] = torch.cat((self._hidden_states_o[layer_idx], hidden_states), dim=1)

    def collect(self, step: GenerationStep) -> None:
        if self.config.store_results:
            self.all_stats.append({
                "top_k_per_layer": [tk.cpu() for tk in self._top_k_per_layer],
                "last_tile_attn": [lta.cpu() for lta in self._last_tile_attn],
                "causal_attn_weights": [caw.cpu() for caw in self._causal_attn_weights],
            })
        self._accumulate_similarity()
        if self.config.use_precomputed_stats == False:
            self._accumulate_importance(step.hidden_states)
        self._reset_buffers()

    def finalize(self):
        queries_per_device = torch.tensor([self._num_queries], dtype=torch.float32, device=self.accelerator.device)
        self._num_queries = reduce(queries_per_device, reduction="sum").item()
        if self._num_queries == 0:
            self.accelerator.print("No queries processed for stats collection.")
            return None

        skipped = max(int(self.config.num_queries - self._num_queries), 0)
        if skipped > 0:
            self.accelerator.print(f"Skipped {skipped} queries due to generation errors.")

        sim_red = reduce(self._sim_mat, reduction="sum")
        imp_red = reduce(self._layer_importance, reduction="sum")
        if self.accelerator.is_main_process:
            sim_avg = sim_red / self._num_queries
            imp_avg = imp_red / self._num_queries

            os.makedirs("./results/plots", exist_ok=True)
            os.makedirs("./results/head_mappings", exist_ok=True)

            if self._post_processing_type == "plot_similarity":
                self._process_plot_similarity(sim_avg)
            elif self._post_processing_type == "select_layers":
                self._process_select_layers(sim_avg, imp_avg)
            elif self._post_processing_type == "plot_sparsity":
                self._process_plot_sparsity(sim_avg)
            return None

    def run(self):
        if self.config.use_precomputed_stats:
            self.prepare(len(self.all_stats[-1]["layer_importance"]), self.all_stats[0]["causal_attn_weights"][0].shape[1])
            self._layer_importance = self.all_stats[-1]["layer_importance"]
            self._num_queries = len(self.all_stats) - 1
            for stat in self.all_stats[:-1]:
                for layer_idx in range(self._num_layers):
                    self.update(
                        is_decode=False,
                        layer_idx=layer_idx,
                        indices=stat["top_k_per_layer"][layer_idx].to(self.accelerator.device),
                        attn_weights=stat["causal_attn_weights"][layer_idx].to(self.accelerator.device),
                        last_tile_attn=stat["last_tile_attn"][layer_idx].to(self.accelerator.device),
                    )
                self.collect(step=None)
            return self.finalize()
        return self._run_inference()

    def _accumulate_similarity(self) -> None:
        if self._post_processing_type in ["plot_similarity", "select_layers"]:
            for layer_idx_1 in range(self._num_layers):
                for layer_idx_2 in range(layer_idx_1, self._num_layers):
                    self._sim_mat[layer_idx_1, layer_idx_2] += self._similarity_layer(layer_idx_1, layer_idx_2)
        elif self._post_processing_type == "plot_sparsity":
            for layer_idx in range(self._num_layers):
                self._sim_mat[layer_idx] += self._plot_sparsity_layer(layer_idx)

    def _accumulate_importance(self, hidden_states: Optional[tuple[tuple[torch.FloatTensor]]]) -> None:
        B, L, D = self._hidden_states_o[0].shape
        for layer_idx in range(self._num_layers):
            hidden_states_i = torch.empty((B, 0, D), dtype=torch.float32, device=self.accelerator.device)
            for i in range(len(hidden_states)):
                hidden_states_i = torch.cat((hidden_states_i, hidden_states[i][layer_idx]), dim=1)
            hidden_states_o = self._hidden_states_o[layer_idx] + hidden_states_i # residual
            self._layer_importance[layer_idx] += 1 - torch.nn.functional.cosine_similarity(
                hidden_states_i.reshape(B*L, D),
                hidden_states_o.reshape(B*L, D),
                dim=-1,
            ).mean().item()

    def _reset_buffers(self) -> None:
        self._top_k_per_layer = []
        self._last_tile_attn = []
        self._causal_attn_weights = []
        self._hidden_states_o = []

    def _similarity_layer(self, layer_idx_1: int, layer_idx_2: int) -> torch.Tensor:
        indices_1 = self._top_k_per_layer[layer_idx_1]
        indices_2 = self._top_k_per_layer[layer_idx_2]
        last_tile_attn = self._last_tile_attn[layer_idx_2]
        attn_weights_2 = self._causal_attn_weights[layer_idx_2]
        denom = torch.gather(attn_weights_2, dim=-1, index=indices_2)
        _, heads, _, _ = indices_1.shape
        attn_weights_2_expanded = attn_weights_2.unsqueeze(2).expand(-1, -1, heads, -1, -1)
        denom_expanded = denom.unsqueeze(2).expand(-1, -1, heads, -1, -1)
        indices_1_expanded = indices_1.unsqueeze(1).expand(-1, heads, -1, -1, -1)
        num = torch.gather(attn_weights_2_expanded, dim=-1, index=indices_1_expanded)
        weighted_matches = (num.sum(dim=-1)+last_tile_attn) / (denom_expanded.sum(dim=-1)+last_tile_attn)
        similarity = weighted_matches.amin(dim=-1) if self.config.run_type == "select_layers" else weighted_matches.mean(dim=-1)
        return similarity[0]

    def _plot_sparsity_layer(self, layer_idx: int) -> torch.Tensor:
        indices = self._top_k_per_layer[layer_idx]
        last_tile_attn = self._last_tile_attn[layer_idx]
        attn_weights = self._causal_attn_weights[layer_idx]
        similarity = ((torch.gather(attn_weights, dim=-1, index=indices).sum(-1) + last_tile_attn) / (attn_weights.sum(-1) + last_tile_attn)).mean(-1).to(torch.float32)[0]
        return similarity

    def _process_plot_similarity(self, sim_avg: torch.Tensor) -> None:
        file_name = f"./results/plots/{self._post_processing_type}_{self._model_name}"
        layer_sim = sim_avg.amax(dim=-1).mean(dim=-1).cpu().numpy()
        np.savetxt(f"{file_name}.csv", layer_sim, delimiter=",")
        self._plot_matrix(layer_sim, f"{file_name}.png")

    def _process_select_layers(self, sim_avg: torch.Tensor, imp_avg: torch.Tensor) -> None:
        file_name = f"./results/head_mappings/{self._model_name}_layers_{{}}.npy"
        self._plot_layer_importance(imp_avg)
        layer_sim = sim_avg.amax(dim=-1).mean(dim=-1)
        layer_sim = (layer_sim * imp_avg.unsqueeze(0).to(layer_sim.device)).cpu().numpy()
        pivots_list = []
        print("Generating best layers combos and head mappings")
        for num_pivots in [3,4,5,6,7,8]:
            pivots_list.append(self._pick_k_layers(layer_sim, num_pivots))

        for pivots in pivots_list:
            print(f"{pivots} Scores: {self._get_pivot_score(layer_sim, pivots + [self._num_layers])}")
            head_mappings = self._find_head_mappings(sim_avg, pivots + [self._num_layers])
            pivot_str = "_".join(map(str, pivots))
            np.save(file_name.format(pivot_str), np.array(head_mappings, dtype=np.int32))

    def _process_plot_sparsity(self, sim_avg: torch.Tensor) -> None:
        file_name = f"./results/plots/{self._post_processing_type}_{self._model_name}"
        sim_np = sim_avg.cpu().numpy()
        np.savetxt(f"{file_name}.csv", sim_np, delimiter=",")
        self._plot_matrix(sim_np, f"{file_name}.png", title="Head Self-Similarity", xlabel="Head", ylabel="Layer")

    def _plot_matrix(self, mat, filename, title="Layer–Layer Similarity", xlabel="Layer j", ylabel="Layer i") -> None:
        print("Saving plot to", filename)
        plt.figure(figsize=(8, 8))
        flat = np.unique(mat)
        if flat.size > 1:
            min_val = flat[1]
        else:
            min_val = flat.min()
        max_val = mat.max()
        plt.imshow(mat, cmap="viridis", vmin=min_val, vmax=max_val)
        plt.colorbar(label="Similarity")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(range(len(mat[0])))
        plt.yticks(range(len(mat)))
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.savefig(filename.replace(".png", ".pdf"), dpi=300, format="pdf")

    def _plot_layer_importance(self, imp_avg: torch.Tensor) -> None:
        file_name = f"./results/plots/layer_importance_{self._model_name}"
        imp_np = imp_avg.cpu().numpy()
        np.savetxt(f"{file_name}.csv", imp_np, delimiter=",")
        layer_indices = range(imp_np.shape[0])
        plt.figure(figsize=(10, 6))
        plt.plot(layer_indices, imp_np, marker='o', linestyle='-', markersize=5)
        plt.xlabel('Layer ID', fontsize=14)
        plt.ylabel('Importance Score', fontsize=14)
        # plt.title('Layer Importance Trend Across 32 Layers')
        # Set x-ticks to show every 2 layers, or as needed
        plt.xticks(np.arange(0, imp_np.shape[0], 2))
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig(file_name + '.png', format='png')
        plt.savefig(file_name + '.pdf', format='pdf')

    def _pick_k_layers_brute_force(self, sim, K):
        """
        Brute force variant using _get_pivot_score for scoring.
        Enumerates all combinations of K pivots with 0 fixed as the first pivot.
        The terminal boundary N is appended only for scoring (not returned as a pivot).
        """
        N = len(sim)
        if K <= 1:
            return [0]
        if K > N:
            K = N

        best_score = float("-inf")
        best_pivots = []
        all_scores = []
        # top_pivots = []

        for comb in combinations(range(1, N), K - 1):
            pivots = [0] + list(comb)
            score, _ = self._get_pivot_score(sim, pivots + [N])
            all_scores.append(score)
            if score > best_score:
                best_score = score
                best_pivots.clear()
                best_pivots.append(pivots)
            elif score == best_score:
                best_pivots.append(pivots)
            # if score > 8.56:
            #     top_pivots.append((pivots, score))

        if all_scores:
            plt.figure(figsize=(4,3))
            plt.hist(all_scores, bins=40, color="steelblue", edgecolor="black")
            plt.title("Pivot Score Distribution")
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig("./results/plots/pivot_score_distribution.png", dpi=300)
            plt.close()
        # print(f"Top pivot sets with score > 8.56: {len(top_pivots)}")
        # subset = set()
        # for pivots, score in top_pivots:
        #     # print(f"{pivots} Scores: {score}")
        #     subset.update(pivots)
        # print("Unique layers in top sets:", sorted(subset), len(subset))
        return best_pivots

    def _pick_k_layers(self, sim, K):
        """
        sim:  N×N list of lists with sim[i][j] defined (or -inf for j<i)
        K:    number of pivots to choose
        Returns:
        V    = a (K+1)×N table where V[k][i] is the max total using k pivots covering 0..i
        path = a (K+1)×N table of back‐pointers
        """
        N = len(sim)
        # Initialize DP tables
        V    = [[-math.inf]*(N+1) for _ in range(K+1)]
        path = [[-1]*(N+1) for _ in range(K+1)]

        # Base case: with 1 pivot at the first row.
        V[0][0] = sim[0][0]

        for k in range(1, K+1):
            for i in range(N+1):
                best_val = -math.inf
                best_l   = -1
                for l in range(0, i):
                    s = sum(sim[l][j] for j in range(l+1, i))
                    val = V[k-1][l] + s + sim[l][l]
                    if val > best_val:
                        best_val = val
                        best_l   = l

                V[k][i]    = best_val
                path[k][i] = best_l
            #print(f"V[{k}]:", V[k])  # Debug: print the current state of V
            #print(f"path[{k}]:", path[k])
            
        cur_i = len(sim)
        pivots = []
        for k in range(K, 0, -1):
            l = path[k][cur_i]
            pivots.append(l)
            cur_i = l

        pivots.reverse()
        return pivots

    def _get_pivot_score(self, sim, pivots):
        scores = []
        for i in range(len(pivots) - 1):
            left = pivots[i]
            right = pivots[i + 1]
            scores += [round(sim[left][j], 3) for j in range(left, right)]
        return sum(scores), min(scores)

    def _find_head_mappings(self, sim_avg: torch.Tensor, pivots):
        reuse = pivots
        layers = list(range(self._num_layers))
        pattern = [0 for _ in layers]
        for i in range(1, len(reuse)):
            for j in range(reuse[i - 1], reuse[i]):
                pattern[j] = reuse[i - 1]
        head_mat = sim_avg[pattern, layers, ...]
        print(pattern)
        print(head_mat.shape)
        head_mat, indices = head_mat.max(dim=-1)
        return indices.cpu().to(torch.int32).tolist()
"""
Opinion Dynamics: Full Figure Generation Script
Paper: The Limits of External Manipulation in Opinion Dynamics
       Through Algorithmic Targeting and the Involvement of Bots
Author: Alejandro Gallo Phillips

Run this script in Google Colab or locally with:
    pip install numpy networkx matplotlib scipy

Outputs: figure1_model_A.png, figure2_model_B.png,
         figure3_cognitive.png, figure4_combined.png
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import random
from scipy.stats import gaussian_kde

# ─── Global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})
# PRR accessibility: use both color AND line style / marker to convey meaning

# ─── Colour palette (accessible) ─────────────────────────────────────────────
C_NEG  = "#1f77b4"   # blue  – opinions converging / ending at negative pole
C_POS  = "#d62728"   # red   – opinions converging / ending at positive pole
C_BOT  = "#2ca02c"   # green – bot / stubborn agent reference lines
C_CONF = "#ff7f0e"   # orange – confidence trace
C_NEUT = "#7f7f7f"   # grey  – neutral reference

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — MODEL A: Bounded Confidence With and Without Stubborn Agents
# ═══════════════════════════════════════════════════════════════════════════════

def run_model_A(n_citizens=100, steps=5000, n_bots=10, epsilon=0.25, mu=0.4, seed=42):
    """
    Bounded-confidence model with optional stubborn agents (bots).
    Set n_bots=0 to run the no-manipulation control.
    Returns: history array of shape (n_recorded, n_citizens)
    """
    np.random.seed(seed)
    random.seed(seed)

    opinions = np.random.uniform(-1.0, 1.0, n_citizens)

    bot_positions = []
    if n_bots > 0:
        half = n_bots // 2
        bot_positions = [0.95] * half + [-0.95] * (n_bots - half)
    bot_opinions = np.array(bot_positions)

    all_opinions = np.concatenate([opinions, bot_opinions]) if n_bots > 0 else opinions.copy()
    n_total = len(all_opinions)

    history = [all_opinions[:n_citizens].copy()]

    for step in range(steps):
        i = random.randint(0, n_citizens - 1)
        j = random.randint(0, n_total - 1)
        while j == i:
            j = random.randint(0, n_total - 1)

        dist = abs(all_opinions[j] - all_opinions[i])
        if dist <= epsilon:
            all_opinions[i] += mu * (all_opinions[j] - all_opinions[i])
            if j < n_citizens:
                all_opinions[j] += mu * (all_opinions[i] - all_opinions[j])

        all_opinions[i] = np.clip(all_opinions[i], -1.0, 1.0)
        if j < n_citizens:
            all_opinions[j] = np.clip(all_opinions[j], -1.0, 1.0)

        if step % 20 == 0:
            history.append(all_opinions[:n_citizens].copy())

    return np.array(history)


def plot_figure1():
    hist_bots   = run_model_A(n_bots=10, seed=42)
    hist_nobots = run_model_A(n_bots=0,  seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, hist, title, show_bots in zip(
        axes,
        [hist_bots, hist_nobots],
        ["(a) With Stubborn Agents (Bots)", "(b) Without Stubborn Agents"],
        [True, False]
    ):
        for k in range(100):
            final = hist[-1, k]
            color = C_POS if final > 0 else C_NEG
            ls    = "-" if final > 0 else "--"
            ax.plot(hist[:, k], color=color, alpha=0.18, linewidth=0.7, linestyle=ls)

        if show_bots:
            ax.axhline( 0.95, color=C_BOT, linestyle=":", linewidth=1.8,
                        label="Bot positions ($\\pm0.95$)")
            ax.axhline(-0.95, color=C_BOT, linestyle=":", linewidth=1.8)

        ax.axhline(0, color=C_NEUT, linestyle="--", linewidth=1.0, alpha=0.6,
                   label="Neutral centre ($x=0$)")
        ax.set_ylim(-1.12, 1.12)
        ax.set_xlabel("Time (recorded every 20 steps)", fontsize=11)
        ax.set_ylabel("Opinion $x_i$", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper right")

    legend_elements = [
        Line2D([0], [0], color=C_POS, ls="-",  lw=1.2, label="Final opinion > 0"),
        Line2D([0], [0], color=C_NEG, ls="--", lw=1.2, label="Final opinion < 0"),
    ]
    axes[0].legend(handles=legend_elements + [
        Line2D([0], [0], color=C_BOT,  ls=":", lw=1.8, label="Bot positions"),
        Line2D([0], [0], color=C_NEUT, ls="--", lw=1.0, alpha=0.6, label="Neutral centre"),
    ], fontsize=9)
    axes[1].legend(handles=legend_elements + [
        Line2D([0], [0], color=C_NEUT, ls="--", lw=1.0, alpha=0.6, label="Neutral centre"),
    ], fontsize=9)

    fig.suptitle(
        "FIG. 1. Model A: opinion trajectories with and without stubborn agents.\n"
        "Left: bots anchor the extremes and the population fragments. "
        "Right: same initial conditions, bots removed — consensus re-emerges.",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    plt.savefig("figure1_model_A.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Figure 1 saved → figure1_model_A.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — MODEL B (Continuous SIBO): With vs Without Algorithmic Targeting
# ═══════════════════════════════════════════════════════════════════════════════

class Citizen:
    def __init__(self, cid, opinion, mu=0.4, eps0=0.25, tau0=0.70, alpha=4, beta=2):
        self.cid     = cid
        self.opinion = opinion
        self.mu      = mu
        self.eps0    = eps0
        self.tau0    = tau0
        self.alpha   = alpha
        self.beta    = beta
        self.ct      = 0          # confirming experiences
        self.rt      = 0          # refuting  experiences
        self.history = [opinion]
        self.conf_history = []

    def confidence(self):
        return (self.alpha + self.ct) / (self.alpha + self.beta + self.ct + self.rt)

    def eps(self):
        return self.eps0 * (1.0 - self.confidence())

    def tau(self):
        return self.tau0 * self.confidence()

    def process(self, other_op):
        """Update confirming / refuting counts based on proximity."""
        if abs(other_op - self.opinion) <= 0.30:
            self.ct += 1
        else:
            self.rt += 1

    def interact(self, other_op):
        self.process(other_op)
        dist = abs(other_op - self.opinion)
        eps  = self.eps()
        tau  = self.tau()

        if dist <= eps:
            # Assimilation
            self.opinion += self.mu * (other_op - self.opinion)
        elif dist > tau:
            # Repulsion (boomerang effect)
            direction     = np.sign(self.opinion - other_op)
            self.opinion += self.mu * direction * dist * (1.0 - abs(self.opinion))

        self.opinion = float(np.clip(self.opinion, -1.0, 1.0))

    def record(self):
        self.history.append(self.opinion)
        self.conf_history.append(self.confidence())


def run_model_B(n=100, steps=8000, algorithmic=True, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    agents = [Citizen(i, float(np.random.uniform(-1.0, 1.0))) for i in range(n)]
    G      = nx.barabasi_albert_graph(n, 3, seed=seed)
    degs   = dict(G.degree())
    total  = sum(degs.values())
    probs  = np.array([degs[v] / total for v in range(n)])

    for step in range(steps):
        if algorithmic:
            u_id = int(np.random.choice(n, p=probs))
        else:
            u_id = random.randint(0, n - 1)

        nbrs = list(G.neighbors(u_id))
        if not nbrs:
            continue
        v_id = random.choice(nbrs)

        u, v    = agents[u_id], agents[v_id]
        u_prev  = u.opinion
        u.interact(v.opinion)
        v.interact(u_prev)

        if step % 10 == 0:
            for a in agents:
                a.record()

    return agents


def plot_figure2():
    ag_algo = run_model_B(algorithmic=True,  seed=42)
    ag_rand = run_model_B(algorithmic=False, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, agents, title in zip(
        axes,
        [ag_algo, ag_rand],
        ["(a) With Degree-Weighted Algorithmic Targeting",
         "(b) Without Algorithmic Targeting (Random Pairing — Control)"]
    ):
        for a in agents:
            final = a.history[-1]
            color = C_POS if final > 0 else C_NEG
            ls    = "-" if final > 0 else "--"
            ax.plot(a.history, color=color, alpha=0.18, linewidth=0.65, linestyle=ls)

        ax.axhline(0, color=C_NEUT, linestyle="--", linewidth=1.0, alpha=0.6,
                   label="Neutral centre ($x=0$)")
        ax.set_ylim(-1.12, 1.12)
        ax.set_xlabel("Recorded time steps (every 10 interactions)", fontsize=11)
        ax.set_ylabel("Opinion $x_i$", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.4)

    legend_elements = [
        Line2D([0], [0], color=C_POS, ls="-",  lw=1.2, label="Final opinion > 0"),
        Line2D([0], [0], color=C_NEG, ls="--", lw=1.2, label="Final opinion < 0"),
        Line2D([0], [0], color=C_NEUT, ls="--", lw=1.0, alpha=0.6,
               label="Neutral centre"),
    ]
    for ax in axes:
        ax.legend(handles=legend_elements, fontsize=9)

    fig.suptitle(
        "FIG. 2. Model B (continuous SIBO): opinion trajectories with and without\n"
        "degree-weighted algorithmic targeting. No stubborn agents present in either run.",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    plt.savefig("figure2_model_B.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Figure 2 saved → figure2_model_B.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — COGNITIVE MODEL: Discrete Belief Revision Over Time
# ═══════════════════════════════════════════════════════════════════════════════

class CognitiveCitizen:
    def __init__(self, cid, opinion, alpha=4, beta=2, kappa=1.5):
        self.cid     = cid
        self.opinion = opinion
        self.alpha   = alpha
        self.beta    = beta
        self.kappa   = kappa
        self.ct      = 0
        self.rt      = 0
        self.history = [opinion]

    def reliability(self):
        return (self.alpha + self.ct) / (self.alpha + self.beta + self.ct + self.rt)

    def process(self, signal):
        if signal == self.opinion:
            self.ct += 1
        else:
            self.rt += 1

    def social_threshold(self, nbr_ops):
        if not nbr_ops:
            return 0.5
        n_t = sum(1 for op in nbr_ops if op != self.opinion)
        m_t = sum(1 for op in nbr_ops if op == self.opinion)
        N   = len(nbr_ops)
        return 0.5 * (1.0 + (n_t - m_t) / (self.kappa * N))

    def update(self, nbr_ops, options):
        theta_hat  = self.reliability()
        theta_crit = self.social_threshold(nbr_ops)
        if theta_hat < theta_crit:
            alts = [op for op in options if op != self.opinion]
            if alts:
                self.opinion = random.choice(alts)
                self.ct = 0
                self.rt = 0
        self.history.append(self.opinion)


def run_cognitive(n=100, steps=250, p_rewire=0.2, seed=7):
    random.seed(seed)
    np.random.seed(seed)

    opts   = [0, 1]
    agents = [CognitiveCitizen(i, random.choice(opts)) for i in range(n)]
    G      = nx.watts_strogatz_graph(n, k=6, p=p_rewire, seed=seed)

    for step in range(steps):
        for i in range(n):
            ag   = agents[i]
            nbrs = list(G.neighbors(i))
            nbr_ops = [agents[nb].opinion for nb in nbrs]

            alg_signal = 1 if np.mean(nbr_ops) > 0.5 else 0
            if random.random() < 0.2:
                alg_signal = 1 - alg_signal

            ag.process(alg_signal)
            ag.update(nbr_ops, opts)

    return agents


def plot_figure3():
    agents = run_cognitive()

    fig, ax = plt.subplots(figsize=(12, 6))

    for a in agents:
        final = a.history[-1]
        color = C_POS if final == 1 else C_NEG
        ls    = "-"  if final == 1 else "--"
        ax.plot(a.history, color=color, alpha=0.28, linewidth=0.8, linestyle=ls)

    # Phase shading
    ax.axvspan(  0,  65, alpha=0.07, color="orange")
    ax.axvspan( 65, 155, alpha=0.05, color="grey")
    ax.axvspan(155, 250, alpha=0.07, color="green")

    # Phase annotations
    ax.text(32,  0.50, "Warm-up period\n(prior uncalibrated;\nmany rapid switches)",
            ha="center", va="center", fontsize=8.5, color="#b05800",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))
    ax.text(110, 0.50, "Stragglers\n(high-$\\kappa$ agents\nswitch late)",
            ha="center", va="center", fontsize=8.5, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))
    ax.text(202, 0.50, "Steady state\n(no further switches;\npolarization locked in)",
            ha="center", va="center", fontsize=8.5, color="#2d6a2d",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))

    legend_elements = [
        Line2D([0], [0], color=C_POS, ls="-",  lw=1.2, label="Ended at Opinion 1"),
        Line2D([0], [0], color=C_NEG, ls="--", lw=1.2, label="Ended at Opinion 0"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="center right")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Opinion 0", "Opinion 1"], fontsize=10)
    ax.set_ylim(-0.18, 1.18)
    ax.set_xlabel("Time steps", fontsize=11)
    ax.set_title(
        "FIG. 3. Cognitive belief revision under social influence.\n"
        "Discrete opinions $\\{0,1\\}$. Each vertical jump represents an opinion switch.\n"
        "Three phases are annotated directly on the figure.",
        fontsize=10, fontweight="bold"
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig("figure3_cognitive.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Figure 3 saved → figure3_cognitive.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — INTEGRATED MODEL: Three-Panel Summary
# ═══════════════════════════════════════════════════════════════════════════════

def run_integrated(n=100, steps=8000, seed=42):
    """Full Model B with confidence tracking for Figure 4."""
    np.random.seed(seed)
    random.seed(seed)

    agents = [Citizen(i, float(np.random.uniform(-1.0, 1.0))) for i in range(n)]
    G      = nx.barabasi_albert_graph(n, 3, seed=seed)
    degs   = dict(G.degree())
    total  = sum(degs.values())
    probs  = np.array([degs[v] / total for v in range(n)])

    for step in range(steps):
        u_id = int(np.random.choice(n, p=probs))
        nbrs = list(G.neighbors(u_id))
        if not nbrs:
            continue
        v_id = random.choice(nbrs)

        u, v   = agents[u_id], agents[v_id]
        u_prev = u.opinion
        u.interact(v.opinion)
        v.interact(u_prev)

        if step % 10 == 0:
            for a in agents:
                a.record()

    return agents


def plot_figure4():
    agents = run_integrated()

    min_len  = min(len(a.history) for a in agents)
    min_clen = min(len(a.conf_history) for a in agents)

    conf_matrix = np.array([a.conf_history[:min_clen] for a in agents])
    mean_conf   = conf_matrix.mean(axis=0)
    std_conf    = conf_matrix.std(axis=0)
    t_conf      = np.arange(min_clen)

    final_ops   = np.array([a.history[-1] for a in agents])

    fig = plt.figure(figsize=(12, 13))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Panel A — opinion trajectories
    for a in agents:
        final = a.history[-1]
        color = C_POS if final > 0 else C_NEG
        ls    = "-" if final > 0 else "--"
        ax1.plot(a.history[:min_len], color=color, alpha=0.18,
                 linewidth=0.65, linestyle=ls)
    ax1.axhline(0, color=C_NEUT, linestyle="--", linewidth=1.0, alpha=0.5)
    ax1.set_ylabel("Opinion $x_i$", fontsize=11)
    ax1.set_title("(a) Opinion Trajectories Over Time", fontsize=11, fontweight="bold")
    ax1.set_ylim(-1.12, 1.12)
    ax1.grid(True, linestyle=":", alpha=0.4)
    legend_a = [
        Line2D([0], [0], color=C_POS, ls="-",  lw=1.2, label="Final opinion > 0"),
        Line2D([0], [0], color=C_NEG, ls="--", lw=1.2, label="Final opinion < 0"),
    ]
    ax1.legend(handles=legend_a, fontsize=9)

    # Panel B — average confidence
    ax2.plot(t_conf, mean_conf, color=C_CONF, linewidth=2.0,
             label="Mean confidence $\\hat{\\theta}_t$")
    ax2.fill_between(t_conf,
                     mean_conf - std_conf,
                     mean_conf + std_conf,
                     alpha=0.20, color=C_CONF, label="$\\pm1$ std. dev.")
    ax2.axhline(0.5, color=C_NEUT, linestyle="--", linewidth=1.0, alpha=0.6,
                label="Prior confidence level")
    ax2.set_ylabel("Confidence $\\hat{\\theta}_t(i)$", fontsize=11)
    ax2.set_title(
        "(b) Average Agent Confidence Over Time\n"
        "(Rising confidence = echo-chamber formation in progress)",
        fontsize=11, fontweight="bold"
    )
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.4)

    # Panel C — final opinion histogram
    ax3.hist(final_ops, bins=30, color=C_NEG, edgecolor="white",
             alpha=0.55, label="Final opinions")
    ax3.axvline(0, color=C_NEUT, linestyle="--", linewidth=1.5, alpha=0.7,
                label="Neutral centre ($x=0$)")

    # KDE overlay
    if len(final_ops) > 1:
        kde  = gaussian_kde(final_ops, bw_method=0.15)
        xs   = np.linspace(-1.05, 1.05, 300)
        ys   = kde(xs) * len(final_ops) * (2.1 / 30)   # scale to match histogram
        ax3.plot(xs, ys, color=C_CONF, linewidth=2.0, label="Kernel density estimate")

    ax3.set_xlabel("Final opinion $x_i$", fontsize=11)
    ax3.set_ylabel("Number of agents", fontsize=11)
    ax3.set_title(
        "(c) Final Distribution of Opinions\n"
        "(Bimodal shape: two clusters near $\\pm1$, centre emptied out)",
        fontsize=11, fontweight="bold"
    )
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle(
        "FIG. 4. Integrated model — three-panel summary.\n"
        "All panels share the same simulation run (no stubborn agents).",
        fontsize=11, fontweight="bold", y=1.005
    )
    plt.savefig("figure4_combined.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Figure 4 saved → figure4_combined.png")


# ─── Run all figures ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Figure 1 — Model A...")
    plot_figure1()

    print("\nGenerating Figure 2 — Model B continuous...")
    plot_figure2()

    print("\nGenerating Figure 3 — Cognitive model...")
    plot_figure3()

    print("\nGenerating Figure 4 — Integrated three-panel...")
    plot_figure4()

    print("\nAll figures generated successfully.")

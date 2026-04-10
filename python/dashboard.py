import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.use("TkAgg")   

class Dashboard:
    def __init__(self, enabled = True):
        self.enabled  = enabled
        self.fig = None
        self._im = None    
        self._ax_game = None
        self._data = {
            k: [] for k in ["xs", "ret", "aloss", "closs", "ent", "sps"]
        }
        if not enabled:
            return
        try:

            self._plt = plt
            self.fig = plt.figure(figsize=(17, 7))
            self.fig.suptitle(
                "MAPPO — Overcooked-AI partial obs sim2real",
                fontsize=12, fontweight="bold",
            )

            # Outer: metrics left (3/5 width) + game render right (2/5 width)
            outer = gridspec.GridSpec(
                1, 2, figure=self.fig,
                width_ratios=[3, 2], hspace=0.05, wspace=0.30,
            )
            # Metrics: 2×3 grid in left column
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 3, subplot_spec=outer[0],
                hspace=0.55, wspace=0.40,
            )

            axes = {
                "ret": self.fig.add_subplot(inner[0, :2]),   # wide top-left
                "aloss": self.fig.add_subplot(inner[0, 2]),
                "closs": self.fig.add_subplot(inner[1, 0]),
                "ent": self.fig.add_subplot(inner[1, 1]),
                "sps": self.fig.add_subplot(inner[1, 2]),
            }
            titles = {
                "ret":   ("Mean episode return (100-ep window)", "return", "b"),
                "aloss": ("Actor loss", "loss",   "r"),
                "closs": ("Critic loss", "loss",   "g"),
                "ent":   ("Policy entropy", "nats",   "m"),
                "sps":   ("Steps / second", "sps",    "c"),
            }
            self._lines = {}
            for key, (title, ylabel, color) in titles.items():
                ax = axes[key]
                ax.set_title(title, fontsize=9)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.set_xlabel("update", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.25)
                self._lines[key], = ax.plot([], [], color=color, lw=1.5)
            self._axes = axes

            # Game render panel — right column, spans both rows
            self._ax_game = self.fig.add_subplot(outer[1])
            self._ax_game.set_title("Eval rollout (no noise)", fontsize=10)
            self._ax_game.axis("off")
            # Placeholder grey image so the panel fills immediately
            placeholder = np.full((300, 400, 3), 40, dtype=np.uint8)
            self._im = self._ax_game.imshow(placeholder, aspect="auto")
            self._ax_game.text(
                0.5, 0.5, "waiting for first eval…",
                transform=self._ax_game.transAxes,
                ha="center", va="center",
                fontsize=9, color="white", alpha=0.7,
            )

            plt.ion()
            plt.show(block=False)

        except Exception as exc:
            print(f"[Dashboard] disabled ({exc})")
            self.fig      = None
            self._ax_game = None
            self.enabled  = False

    def update(self, update_idx, mean_ret, actor_loss, critic_loss, entropy, sps):
        """Update metric plots. Call every log_interval updates."""
        if not self.enabled or self.fig is None:
            return
        d = self._data
        d["xs"].append(update_idx)
        d["ret"].append(mean_ret)
        d["aloss"].append(actor_loss)
        d["closs"].append(critic_loss)
        d["ent"].append(entropy)
        d["sps"].append(sps)
        try:
            for key in ["ret", "aloss", "closs", "ent", "sps"]:
                self._lines[key].set_xdata(d["xs"])
                self._lines[key].set_ydata(d[key])
                self._axes[key].relim()
                self._axes[key].autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def update_render(self, frame, step, ep_return):
        if not self.enabled or self._ax_game is None or self._im is None:
            return
        try:
            self._im.set_data(frame)
            self._im.set_extent([0, frame.shape[1], frame.shape[0], 0])
            self._ax_game.set_title(
                f"Eval — step {step}   ret {ep_return:.1f}",
                fontsize=9,
            )
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def save(self, path):
        if self.fig is not None:
            try:
                self.fig.savefig(path, dpi=150, bbox_inches="tight")
                print(f"  Dashboard saved: {path}")
            except Exception:
                pass
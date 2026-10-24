"""Terminal reward of the legacy RL server: outcome (+1 / -1 / 0) only (plan objective-dojo removed the life-force term);
the opponent label comes from the policy key, else from the harness ("op"; advanced -> "bot")."""
import queue
import types
from collections import Counter

from swccg_policy.rl import RLServer


def server():
    s = RLServer.__new__(RLServer)
    s.a = types.SimpleNamespace()
    s.traj, s.game_keys, s.game_snap, s.out_q, s.stats = {}, {}, {}, queue.Queue(), Counter()
    s.slots = {"learner": types.SimpleNamespace(name="learner")}
    return s


def end(s, outcome, winner, lf=None, op=None):
    s.traj[("g", "ds")] = [{"rec": [], "seq": [0], "logp": 0.0}]
    s.game_keys["g"] = {"ds": "learner"}
    env = {"end": "g", "s": "ds", "o": outcome, "w": winner}
    if lf is not None:
        env["lf"] = lf
    if op is not None:
        env["op"] = op
    s.finish(env)
    return s.out_q.get_nowait()


def test_reward():
    s = server()
    g = end(s, "finished", "ds", [30, 0], "random")
    assert g["reward"] == 1.0 and g["won"] == 1 and g["opp"] == "random"
    g = end(s, "finished", "ls", [0, 25], "advanced")
    assert g["reward"] == -1.0 and g["won"] == 0 and g["opp"] == "bot"
    g = end(s, "cap", "", [12, 5])
    assert g["reward"] == 0.0 and g["won"] == 0
    g = end(s, "finished", "ds")
    assert g["reward"] == 1.0


if __name__ == "__main__":
    test_reward()
    print("ok")

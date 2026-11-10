# Scenarios and experiments (Steven)

Kept outside `theory/` on purpose: `sync_theory.sh` mirrors Eeshan's notes folder with `--delete`,
so anything uploaded straight into `theory/` is removed at the next sync.

## What is in this upload

| Path | What it does | Run it |
|---|---|---|
| `simlab/tictactoe.py` | Tic-tac-toe world: 9 squares (K = 9), 10 features per square, minimax checks | (library) |
| `experiments/exp02_tictactoe.py` | 300 simulated beginners per policy; training, then graduation | `python -m experiments.exp02_tictactoe` (about 20 s) |
| `experiments/play_tictactoe.py` | Play it yourself: pre-test, training with probes, graduation | `python -m experiments.play_tictactoe --hints conflict` |
| `experiments/exp03_duolingo_curves.py` | Fits the simlab learning curves and a forgetting half-life to Duolingo data | `python -m experiments.exp03_duolingo_curves learning_traces.13m.csv.gz` |
| `experiments/exp04_weather.py` | Fits the weather world: truth weights, symptom strength, value of each test | `python -m experiments.exp04_weather weatherAUS.csv` |
| `results/exp02/`, `results/RESULTS_exp02_exp03.md` | Figure, numbers and write-up for tic-tac-toe | |

Needs `numpy scipy matplotlib pandas scikit-learn`. The Duolingo and weather data are not in the repo:
Duolingo's file is on Harvard Dataverse (linked from github.com/duolingo/halflife-regression); weatherAUS
is the "Rain in Australia" dataset (Kaggle, or R's `rattle` package).

## Hint styles in the play script (learning vs. following)

The first version's hints named the square ("square 9 blocks their win"), so players followed them and
learned nothing. `--hints` now offers:

- `presence` (default): says what to look for, not where.
- `conflict`: shows features on positions where they point to different squares, so the player must weigh them.
  Closest to the model: x_j is shown and the choice depends on the player's weights h.
- `after`: commit first, then every square's features and the best move are revealed.
- `answer`: the old style, kept for comparison.

Every third training move is an unhinted probe. The script reports hinted accuracy, probe accuracy and
the transfer gap (hinted minus probe). psi(m) should be fitted from probes and graduation moves only,
never from hinted moves. Each session is saved to `results/play/` (one row per move).

## Results so far

- **Tic-tac-toe (exp02, simulated):** an untrained beginner checks "two-in-a-row" and "win", never
  "block", and errs on 33% of moves. A planned curriculum reaches the 6.2% floor in 2 training rounds;
  "help now" needs about 20. Crowding-out appears in a real game.
- **Weather (exp04, 58,367 station-days, 26 stations, 2007-2017):** linear truth AUROC 0.883 vs 0.890
  boosted. Symptoms (cloud, today's rain, wind, temperature) explain 70% of the true margin's variance,
  against 6.9% in sepsis, so the lambda > 0 policies have teachable moments to use. Afternoon humidity
  is the single most useful test (70% -> 89%). Morning humidity, evaporation and temperature are
  near-zero-weight distractors.

## Scenario map

| Scenario | Truth U | Symptoms strong? | Human learning data | Status |
|---|---|---|---|---|
| Tic-tac-toe | Exact (minimax) | Yes | Our own play sessions | Built |
| Weather (rain tomorrow) | Logistic fit, AUROC 0.883 | Yes (70%) | Weather prediction task literature | Fitted |
| Sepsis (Chirag) | Logistic fit, AUROC 0.705 | No (6.9%) | None | Built |
| START mass-casualty triage | Exact (protocol) | Yes | Published error rates | Not built |
| Duolingo | n/a | n/a | 13M recall traces: psi and forgetting | Script ready |
| Cyber-security alert triage | Analyst-confirmed labels | Alert summary vs. log lookups | Not checked yet | Idea (Eeshan) |
| Fraud detection (BAF) | Fraud labels | Form fields vs. lookups | None | Idea |
| Legal decisions | Hard to define | Unclear | Unclear | Idea (Eeshan) |
| LLM tool calls under a latency budget | Task success | Prompt vs. tool results | The learner is a model, so curves can be measured directly | Idea (Eeshan) |

**Eeshan's triage point** (train interns to pick the right tests, not "shotgun testing") is what the
graduation phase measures: after training, the learner chooses their own c tests.

**Chirag's point on U** (an unknown U still exists, versus a U that is itself wrong): weather and sepsis
measure how wrong a linear U is (AUROC 0.883 vs 0.890; 0.705 vs 0.784). A clean experiment: teach the
linear U, but score graduation against the boosted model.

**The LLM-tool-call setting** avoids the human-subjects problem: a small model's "beliefs" can be
probed directly, so psi(m) can be measured exactly rather than inferred.

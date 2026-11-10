# Experiments 2 and 3: tic-tac-toe teaching, Duolingo learning curves

## Experiment 2: teaching a novice tic-tac-toe player (`python -m experiments.exp02_tictactoe`)

**World** (`simlab/tictactoe.py`). The 9 squares are the K = 9 actions. Every square gets 10 features
describing what playing there does. Geometry features (occupied, centre, corner, opposite corner, edge)
are always visible: the `s = Bx` part. Tactical features (win, block, fork, block fork, threat) stay hidden
unless the AI reveals them, at most c = 2 per round. The state x stacks the nine feature vectors
(n = 90). U and H are block-diagonal, so `u_k . x = w . f(k)`, and the run uses `simlab.simulate` and
`CountLearner` unchanged.

- **Truth `w`:** the Newell-Simon priority order (win > block > fork > block fork > centre > corners >
  edge), written as weights. It picks a minimax-optimal move in 98.4% of the 3,191 positions where the
  choice matters.
- **Novice `h0`:** likes the centre and making two-in-a-row (threat 3.0 vs. a true 0.5), half-sees her
  own wins (3 vs. 100), and mostly ignores the opponent's (block 0.5 vs. 50).
- **Positions:** all reachable positions where the choice matters. 70% are used for training and 30% are
  held out for graduation.
- **Graduation:** with no hints, she checks the c = 2 tactical features she believes matter most
  (|h_f| x spread of f), then moves. We count moves that lower the minimax value of the game.
- **Policies:** no hints; *help now* (reveal whatever makes this move best, seeing the position);
  *take turns* (rotate through the features); *planned curriculum* (a greedy planner that chooses lessons
  for the graduation objective in advance, without seeing positions).

**Main results**, exponential learning, 5% of the gap closed per lesson, 300 learners per cell:

| Training rounds | No hints | Help now | Take turns | Planned curriculum |
|---|---|---|---|---|
| 2  | 33.5% / 34.8 | 22.8% / 12.6 | 18.6% / 25.3 | **6.2% / 8.7** |
| 5  | 33.5% / 32.8 | 13.1% / 8.1  | 6.2% / 23.2  | **6.2% / 4.2** |
| 12 | 33.2% / 32.7 | 7.4% / 4.8   | 6.2% / 23.4  | **6.2% / 2.2** |
| 30 | 30.5% / 33.4 | 6.2% / 2.4   | 6.2% / 27.0  | **6.2% / 1.4** |

Each cell shows mistakes after graduation (% of moves), then regret per move during training. The floor
with perfect beliefs is 6.2% (two checks can't catch forks). A novice with no training makes mistakes on
33.4% of moves.

**Findings**

1. **Crowding-out shows up.** An untrained novice checks *threat* (her favourite) and *win*, never
   *block*. After 2 rounds of help-now, 95% of learners still check threat. The planned curriculum
   teaches win and block together, which pushes block above threat in her own ranking. Within 2 rounds,
   every learner checks win and block.
2. **Planned is best on both measures.** It reaches the graduation floor in 2 rounds, and it also has the
   lowest training regret from 5 rounds on, below help-now, which only optimizes the current move. Lessons
   that teach early pay off in later training rounds too.
3. **Take turns graduates well but is costly during training.** It hides win and block in most rounds, so
   its training regret stays high (23 to 27).
4. **Rounds until within 1 point of the floor:**

   | Learning curve | Planned | Take turns | Help now |
   |---|---|---|---|
   | exponential 0.05 | 2 | 5 | 20 |
   | exponential 0.2 | 1 | 1 | 12 |
   | power law 0.1* | 3 | 12 | 12 |
   | sigmoid, midpoint 6* | 12 | 12 | 30 |

   *These two curves were only run at 3, 12 and 30 rounds. The planned curriculum is never slower than
   the alternatives. The slow-start sigmoid learner narrows the gap.

**Caveats**

- The learner is simulated. No human tic-tac-toe data is used.
- Positions are drawn uniformly, not by how often they come up in real games.
- The graduation check rule (|h| x spread) is our modelling choice.
- With c = 2, perfect beliefs still miss forks (6.2%); with c = 3 the floor is 1.9%.

Output: `results/exp02/fig1_graduation.png`, `results/exp02/summary.json`.

## Experiment 3: real learning and forgetting curves from Duolingo (`python -m experiments.exp03_duolingo_curves FILE`)

This one needs the data. Duolingo's spaced-repetition file (`learning_traces.13m.csv.gz`, about 361 MB) is
on Harvard Dataverse, linked from github.com/duolingo/halflife-regression. The script streams the file
once and fits two things:

1. **Learning:** mean recall after m exposures (gaps under a day), fit with each `simlab.curves` family
   (exponential, hyperbolic, power law, sigmoid). This tells us which phi real learners follow, and its
   parameter.
2. **Forgetting:** recall against days since last practice, at fixed m, fit as `a * 2^(-days / half_life)`.
   This gives the decay rate the memory-decay extension needs.

The script was tested on a synthetic file in the same format, generated from a known power-law learner
(exponent 0.6) with a 20-day half-life. It picked the power law with exponent 0.60 and recovered a
20.2-20.3 day half-life in every exposure bin. The real file has not been run yet.

Outputs: `results/exp03/fit.json`, `learning_table.csv`, `forgetting_table.csv`, `fig1_duolingo_curves.png`.
Use `--language es` (or fr, de, ...) for one language.

"""What every page of the lab agrees on: the standard figures, the equations behind them, and the benches.

The Overview page is written from these lists, and every bench draws its figures from them, so a figure
keeps its number, its title and its meaning wherever it appears. To add a bench, add it to BENCHES; to
link a derivation once it is written, set that equation's `derivation` to its URL.

Titles and formulas are HTML (entities, <sub>, <sup>, <i>); they are written here and nowhere else.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Equations. `figures` names the standard figures (by key) that draw the quantity.
# `derivation` is None until one is written; then it is the link the Overview shows.
# ---------------------------------------------------------------------------

EQUATIONS = [
    dict(
        id="value", name="What an action is worth",
        html="(<i>Ux</i>)<sub>k</sub> = &Sigma;<sub>j</sub> u<sub>kj</sub> x<sub>j</sub>,"
             "&emsp;x ~ N(0, I)",
        text="The state x has one number per feature. The truth U has one row per action and one column per "
             "feature: u<sub>kj</sub> is the weight of feature j in the value of action k. States are drawn "
             "afresh every round; on the Correlated features bench they come from N(0, &Sigma;).",
        figures=[], derivation=None),
    dict(
        id="choice", name="The human's choice",
        html="&ycirc;<sub>t</sub>(x) = argmax<sub>k &isin; A<sub>t</sub></sub> "
             "&Sigma;<sub>j &isin; F<sub>t</sub></sub> h<sub>kj,t</sub> x<sub>j</sub>",
        text="In round t the algorithm shows the features F<sub>t</sub> and offers the actions A<sub>t</sub>. "
             "The human scores each offered action with their own weights H<sub>t</sub>, using only the shown "
             "features, and picks the highest. Ties are split evenly.",
        figures=["schedule", "choice"], derivation=None),
    dict(
        id="best", name="The best move",
        html="y*(x) = argmax<sub>k</sub> (<i>Ux</i>)<sub>k</sub>",
        text="The best move uses the truth, every feature and every action, whatever the algorithm holds back. "
             "Regret is always measured against it.",
        figures=["choice", "accuracy"], derivation=None),
    dict(
        id="learning", name="How a belief moves",
        html="h<sub>kj,t</sub> = u<sub>kj</sub> + (h<sub>kj,0</sub> &minus; u<sub>kj</sub>) "
             "(1 &minus; &phi;<sub>j</sub>(N<sub>kj</sub>(t)))",
        text="N<sub>kj</sub>(t) counts the rounds before t in which feature j was shown and action k was offered. "
             "A weight moves only in those rounds; held back, it keeps its value and carries on later from "
             "where it stopped. The speed belongs to the feature: every action's weight on feature j follows "
             "the same curve &phi;<sub>j</sub>.",
        figures=["beliefs", "schedule"], derivation=None),
    dict(
        id="curves", name="The learning curves",
        html="exponential&ensp;&phi;(m) = 1 &minus; (1 &minus; r)<sup>m</sup><br>"
             "hyperbolic&ensp;&phi;(m) = m / (m + m<sub>&frac12;</sub>)<br>"
             "power law&ensp;&phi;(m) = 1 &minus; (1 + m)<sup>&minus;a</sup><br>"
             "sigmoid&ensp;&phi;(m) = (s(m) &minus; s(0)) / (1 &minus; s(0))<br>"
             "with&ensp;s(m) = 1 / (1 + e<sup>&minus;5(m &minus; m<sub>0</sub>) / m<sub>0</sub></sup>)",
        text="&phi;(m) is the share of the gap between the first belief and the truth that is closed after m "
             "rounds in use: it starts at 0, never falls, and tends to 1. Each curve has one parameter, set "
             "on the bench: the rate r, the half-life m<sub>&frac12;</sub>, the exponent a, or the midpoint "
             "m<sub>0</sub>.",
        figures=["beliefs"], derivation=None),
    dict(
        id="regret", name="Regret loss of a round",
        html="&#8467;<sub>t</sub> = E<sub>x</sub>[ max<sub>k</sub> (<i>Ux</i>)<sub>k</sub> &minus; "
             "(<i>Ux</i>)<sub>&ycirc;<sub>t</sub>(x)</sub> ]",
        text="What the best move is worth minus what the chosen move is worth, averaged over states. It is zero "
             "in a state where the human picks the best move. With one feature and two actions it is "
             "|&Delta;u| &middot; E|x| = |&Delta;u| &radic;(2/&pi;) on a wrong round; on the other benches it is "
             "the mean over 2,000 probe states.",
        figures=["regret"], derivation=None),
    dict(
        id="objective", name="The objective: discounted regret",
        html="R = &Sigma;<sub>t = 0</sub><sup>T &minus; 1</sup> &delta;<sup>t</sup> &#8467;<sub>t</sub>",
        text="The algorithm's patience &delta; is one number in (0, 1). Near 0 only the first rounds count; "
             "near 1 later rounds count almost as much. This sum is the large number at the top of every "
             "bench and the quantity every ranking table sorts by.",
        figures=["regret"], derivation=None),
    dict(
        id="value-gap", name="Value gap",
        html="g<sub>t</sub> = E<sub>x</sub>[ max<sub>k</sub> (<i>Ux</i>)<sub>k</sub> &minus; "
             "&ucirc;<sub>t</sub>(&ycirc;) ],&emsp;&ucirc;<sub>t</sub>(&ycirc;) = max<sub>k &isin; A<sub>t</sub></sub> "
             "&Sigma;<sub>j &isin; F<sub>t</sub></sub> h<sub>kj,t</sub> x<sub>j</sub>",
        text="What the best move is truly worth minus what the human expects from the move they pick. It "
             "measures how well calibrated the human is, not whether they choose well: it can sit at zero "
             "while the choice is still wrong.",
        figures=["value"], derivation=None),
    dict(
        id="accuracy", name="Share of states with the best move",
        html="a<sub>t</sub> = P<sub>x</sub>[ &ycirc;<sub>t</sub>(x) = y*(x) ]",
        text="With several features the human is no longer simply right or wrong: it depends on the state. "
             "a<sub>t</sub> is the share of states in which the pick is the best move. 1 &minus; a<sub>t</sub> "
             "is the grey behind every figure and the hatched bar at the top of a bench.",
        figures=["accuracy"], derivation=None),
    dict(
        id="flip", name="The flip, with one feature and two actions",
        html="right in round t&ensp;&hArr;&ensp;&phi;(t) &gt; |&Delta;h<sub>0</sub>| / "
             "(|&Delta;h<sub>0</sub>| + |&Delta;u|)",
        text="With one feature the state scales both actions alike, so the human is right in every state or in "
             "none: right exactly when h<sub>1</sub> &minus; h<sub>2</sub> has the sign of &Delta;u = "
             "u<sub>1</sub> &minus; u<sub>2</sub>. A human who starts on the wrong side flips once, when the "
             "curve passes this threshold. &Delta;h<sub>0</sub> is the first belief gap.",
        figures=[], derivation=None),
    dict(
        id="filling-in", name="Filling in a hidden feature, when features are correlated",
        html="x&#770; = P<sub>S</sub> x,&emsp;E[x<sub>hidden</sub> | x<sub>S</sub>] = "
             "&Sigma;<sub>hS</sub> &Sigma;<sub>SS</sub><sup>&minus;1</sup> x<sub>S</sub>",
        text="On the Correlated features bench the human predicts each hidden feature from the shown ones S and "
             "then chooses with the filled-in state, so they act as if their weights were H<sub>t</sub> "
             "P<sub>S</sub>. A filled-in value never corrects the belief about that feature.",
        figures=[], derivation=None),
]

# ---------------------------------------------------------------------------
# The standard figures: on every bench, in this order, under these numbers.
# ---------------------------------------------------------------------------

STANDARD_FIGURES = [
    dict(
        key="schedule", title="What the algorithm shows and offers, round by round",
        draws="One row per feature, then one row per action. A row is inked in the rounds its feature is "
              "shown or its action is offered, and pale where the algorithm holds it back.",
        read="Read down a column for what the human has to work with in that round. A pale stretch is also a "
             "stretch in which nothing is learned about that row.",
        equations=["choice", "learning"]),
    dict(
        key="beliefs", title="What the human believes: each action's weight on each feature",
        draws="One panel per feature, numbered 2.1, 2.2 and so on; with one feature there is one panel. Each "
              "line is one action's believed weight on that feature, the dot is the first belief, and the "
              "dashed line of the same colour is the truth it is heading for.",
        read="A line is faded and flat across the rounds in which its weight is out of use. How fast a line "
             "closes on its dashed truth is the feature's learning curve.",
        equations=["learning", "curves"]),
    dict(
        key="choice", title="The human's choice &ycirc; against the best move y*, for one reference state",
        draws="For one fixed state, the action the human picks in each round as a solid step line and the "
              "best move as a dashed one, with a grey box over the rounds where they differ.",
        read="This is one state's story. Other states can flip at other rounds, which is what Fig. 6 adds up.",
        equations=["choice", "best"]),
    dict(
        key="regret", title="Regret loss each round, before and after discounting",
        draws="The grey line is the expected regret of each round, &#8467;<sub>t</sub>. The ink line weights "
              "it by &delta;<sup>t</sup>.",
        read="The area under the ink line is the objective, the large number at the top of the bench. A "
             "dashed level, where drawn, is the regret that remains once everything in use is learned.",
        equations=["regret", "objective"]),
    dict(
        key="value", title="Value gap: the best move's true value minus what the human expects from their choice",
        draws="One line, g<sub>t</sub>, around a zero line.",
        read="Above zero the human expects too little from the move they pick; below zero, too much. Zero does "
             "not mean the choice is right.",
        equations=["value-gap"]),
    dict(
        key="accuracy", title="How often the human picks the best move",
        draws="The share of states in which the pick is the best move, a<sub>t</sub>, from 0% to 100%. A grey "
              "line, where drawn, is the bench's reference policy.",
        read="A dashed level is where the share is heading once everything in use is learned. With one feature "
             "the share is 0% or 100%: the human is right in every state or in none.",
        equations=["accuracy"]),
]
for number, figure in enumerate(STANDARD_FIGURES, start=1):
    figure["n"] = number

# ---------------------------------------------------------------------------
# The benches. `notes` adds a line under a standard figure's title on that bench; `absent` gives the
# reason a standard figure is not drawn there (it is still listed, with the reason). `own` are the
# figures only that bench has, numbered on from the standard ones.
# ---------------------------------------------------------------------------

BENCHES = [
    dict(
        endpoint="lab", name="1 feature, 2 actions",
        summary="The smallest world. The human is right in every state or in none, so everything comes down "
                "to one flip and when it happens.",
        notes=dict(
            beliefs="One panel: there is one feature, and both actions share its learning curve.",
            choice="For a positive state. For a negative state the choice and the best move both swap.",
            accuracy="0% or 100% here: with one feature the human is right in every state or in none."),
        absent=dict(
            schedule="Not relevant here. There is one feature and two actions and both are in play in every "
                     "round, so the algorithm has nothing to hold back."),
        own=["The flip: when the belief gap lands on the side of the true gap",
             "Every nearby world at once; the ring is the one on the bench"]),
    dict(
        endpoint="worlds", name="More actions and features",
        summary="Up to six actions and six features, with a fixed choice of which features are shown and which "
                "actions are offered. Ranks every fixed subset.",
        notes=dict(
            schedule="The same every round on this bench.",
            choice="For the state with every feature at +1."),
        absent={}, own=[]),
    dict(
        endpoint="timed", name="Timed hiding",
        summary="The same bench, with windows of rounds in which one feature or one action is held back and "
                "then brought back.",
        notes=dict(choice="For the state with every feature at +1."),
        absent={}, own=[]),
    dict(
        endpoint="correlated_page", name="Correlated features",
        summary="Three features that move together. A hidden feature is filled in from the shown ones, and the "
                "algorithm may explore before committing to a subset.",
        notes=dict(
            schedule="Every action is offered in every round; only the features change.",
            choice="For the state with every feature at +1, hidden features filled in from the shown ones.",
            regret="The dashed line is the best fixed subset, discounted."),
        absent={},
        own=["What correlated states look like: 300 of the probe states, two features at a time, coloured by the best move",
             "What the human actually weighs: beliefs once hidden features are filled in, <i>H</i>&nbsp;<i>P</i><sub>S</sub>, "
             "against the best possible with the same features, <i>U</i>&nbsp;<i>P</i><sub>S</sub>",
             "How long to explore: discounted regret against the length of the exploration phase, one line per subset committed to"]),
]

STANDARD = {figure["key"]: figure for figure in STANDARD_FIGURES}
EQUATION = {equation["id"]: equation for equation in EQUATIONS}
for number, equation in enumerate(EQUATIONS, start=1):
    equation["n"] = number
